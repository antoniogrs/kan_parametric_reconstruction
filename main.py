import os
import sys

BASE_DIR = os.path.abspath(os.path.dirname(__file__)) 
LIB_DIR = os.path.join(BASE_DIR, "lib")
UTILS_DIR = os.path.join(BASE_DIR, "utils")
print("Verifica Directory: " + BASE_DIR)

sys.path.append(LIB_DIR)
sys.path.append(UTILS_DIR)

from torch.utils.data import DataLoader
from utils.funzioni_utilita import *
from utils.log_writer import *
from models.modelli import *

def esegui_training_con_modello(classe_modello):
    '''
    Trains a given neural network model on the selected dataset.

    This method initializes a new model of the specified class (`classe_modello`), 
    sets up checkpoint and results directories, initializes an AdamW optimizer,
    and starts a training routine using MSE loss. The routine includes:
    - Model and directory setup for saving results, best checkpoints, and loss curves.
    - Model initialization and GPU preparation.
    - Training and validation across epochs via the `train()` method, using MSE loss (`torch.nn.MSELoss`).
    - Optional logging and loss plotting using the custom made class `LogWriter`
    
    Args:
        classe_modello : class
            The neural network model class to be instantiated and trained.
            Must accept `dropout_rate` and `nome_rete` as initialization arguments.

    Notes:
        Writes training loss, model checkpoints in `.pt`, and result files as `.json`.
        The checkpoint, results, and best model files are saved in `BASE_DIR` within dedicated directories.
        The routine uses an AdamW optimizer and optional scheduler (controlled by `SCHED_STEP` and `SCHED_GAMMA`).
    '''
    NOME_RETE = f"{classe_modello.__name__}" + "_CilindroSpiralato"
    # Check if the model checkpoint exists
    salvataggio_modelli_path = os.path.join(BASE_DIR, 'models')
    nome_cartella = NOME_RETE.rsplit("_", 1)[-1]
    path_cartella_modello = os.path.join(salvataggio_modelli_path, nome_cartella)
    os.path.exists(path_cartella_modello)

    #Path to the folder containing the .json results
    path_cartella_risultati = os.path.join(BASE_DIR, 'results', nome_cartella)

    # Initialize model and optimizer if no checkpoint is found
    print(f"Creazione nuova rete: {NOME_RETE}")
    model = classe_modello(dropout_rate=DROPOUT, nome_rete=NOME_RETE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    starting_epoch = 0 

    criterion = nn.MSELoss()

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    salvataggio_runs_path = os.path.join(BASE_DIR, 'runs')
    path_cartella_runs = os.path.join(salvataggio_runs_path, nome_cartella)
    if not os.path.exists(path_cartella_runs):
        os.makedirs(path_cartella_runs)
    #Folder for TensorBoardX localhost log data
    runs_nome_cartella = os.path.join(path_cartella_runs, NOME_RETE)
    if not os.path.exists(runs_nome_cartella):
        os.makedirs(runs_nome_cartella)

    log_writer = LogWriter(base_folder=runs_nome_cartella, loss_folder_name="losses")
    print(f"🛠️ Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    train_losses, val_losses, epochs = train(
        model, device, train_loader, val_loader,
        criterion, optimizer, log_writer=log_writer, epochs=EPOCHS,
        patience=None, scheduler_step_size=SCHED_STEP, scheduler_gamma=SCHED_GAMMA,
        path=path_cartella_modello, path_json=path_cartella_risultati, model_name=model.name
    )

def esegui_test(classe_modello):
    '''
    Executes the testing routine for a given model class (`classe_modello`), specifically 
    for a "CilindroSpiralato" dataset type.
    
    - Builds the directory structure for checkpoints, results, and best prediction JSON files.
    - Verifies the existence of checkpoint files and result directories, creating them if needed.
    - Loads the best checkpoint (`_best.pt`), restores model and optimizer state, and prepares the model for evaluation.
    - Evaluates the model on Validation and Test sets, computes R² metrics, and saves them.
    - Computes and saves best and worst prediction metrics (`best_mse`, `best_rmse`) using dedicated methods.
    - Processes and saves prediction results for fixed and randomly selected test images.
    '''

    NOME_RETE = f"{classe_modello.__name__}" + "_CilindroSpiralato"

    salvataggio_modelli_path = os.path.join(BASE_DIR, 'models')
    nome_cartella = NOME_RETE.rsplit("_", 1)[-1]
    path_cartella_modello = os.path.join(salvataggio_modelli_path, nome_cartella)
    os.path.exists(path_cartella_modello)

    risultati_path = os.path.join(BASE_DIR, 'results')
    path_cartella_risultati = os.path.join(risultati_path, nome_cartella)

    best_json_path = os.path.join(BASE_DIR, 'best_pred')
    best_json_path_1 = os.path.join(best_json_path, nome_cartella)
    path_cartella_best_json = os.path.join(best_json_path_1, NOME_RETE)
    os.path.exists(best_json_path)
    
    path_label_test = os.path.join(BASE_DIR, 'dataset', dataset_scelto, 'test', 'labels')
    os.path.exists(path_cartella_modello)

    path_cartella_modello_ricaricato = os.path.join(path_cartella_modello, NOME_RETE)
    miglior_modello = f'{NOME_RETE}_best.pt'
    path_modello_ricaricato = os.path.join(path_cartella_modello_ricaricato, miglior_modello)

    risultati_nome_cartella = os.path.join(path_cartella_risultati, NOME_RETE)
    if not os.path.exists(risultati_nome_cartella):
        os.makedirs(risultati_nome_cartella)

    if os.path.exists(path_modello_ricaricato):
        print(f"Caricamento del modello: {path_modello_ricaricato}")
        # Load model and optimizer from checkpoint
        modello_ricaricato = torch.load(path_modello_ricaricato)
        model = classe_modello(dropout_rate=DROPOUT, nome_rete=NOME_RETE).to(device) 
        model.load_state_dict(modello_ricaricato['model_state_dict'])
        # Load optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(modello_ricaricato['optimizer_state_dict'])

        model.eval()

    #Evaluation of the network with metrics such as R² and MSE for test and validation
    metriche_val = calcola_metriche(model, val_loader, device, set_name="Validation")
    save_metriche(metriche_val, risultati_nome_cartella, model.name, nome_str="val")
    metriche_test = calcola_metriche(model, test_loader, device, set_name="Test")
    save_metriche(metriche_test, risultati_nome_cartella, model.name, nome_str="test")

    best_mse, best_rmse = best_pred_Test(model, test_loader, device, json_dir=path_label_test, save_dir=path_cartella_best_json,
                                          model_name=model.name)
    #Useful for having an additional save point.
    r2_scores = {
        "val_r2": metriche_val["R2"],
        "test_r2": metriche_test["R2"],
        "best_mse": best_mse,
        "best_rmse": best_rmse,
    }
    save_r2_scores(r2_scores, risultati_nome_cartella, model.name)
    
    worst_mse, worst_rmse = peggiore_pred_Test(model, test_loader, device, json_dir=path_label_test, save_dir=path_cartella_best_json, model_name=model.name)

def visualizza_confronti_immagini(classe_modello, nome_img, nome_img_peggiore):
    '''
    Loads a trained and evaluated model, and visually compares its 3D object (mesh) predictions with the actual images.

    - Builds the directory structure for model checkpoints, results, and best prediction files.
    - Verifies the existence of checkpoint files and result directories.
    - Loads the best checkpoint (`_best.pt`), restores the model and optimizer state.
    - Displays side-by-side visual comparisons of mesh and image for both the best and worst test cases.
    - Shows 2D and 3D metric comparisons (Hausdorff distance) for the best and worst results.
    '''
    NOME_RETE = f"{classe_modello.__name__}" + "_CilindroSpiralato"

    # Check if the model checkpoint exists
    salvataggio_modelli_path = os.path.join(BASE_DIR, 'models')
    nome_cartella = NOME_RETE.rsplit("_", 1)[-1]
    path_cartella_modello = os.path.join(salvataggio_modelli_path, nome_cartella)
    os.path.exists(path_cartella_modello)

    risultati_path = os.path.join(BASE_DIR, 'results')
    path_cartella_risultati = os.path.join(risultati_path, nome_cartella)

    best_json_path = os.path.join(BASE_DIR, 'best_pred')
    best_json_path_1 = os.path.join(best_json_path, nome_cartella)
    path_cartella_best_json = os.path.join(best_json_path_1, NOME_RETE)
    os.path.exists(best_json_path)
    
    path_label_test = os.path.join(BASE_DIR, 'dataset', dataset_scelto, 'test', 'labels')
    os.path.exists(path_cartella_modello)

    path_cartella_modello_ricaricato = os.path.join(path_cartella_modello, NOME_RETE)
    miglior_modello = f'{NOME_RETE}_best.pt'
    path_modello_ricaricato = os.path.join(path_cartella_modello_ricaricato, miglior_modello)

    if os.path.exists(path_modello_ricaricato):
        print(f"Caricamento del modello: {path_modello_ricaricato}")
        # Load model and optimizer from checkpoint
        modello_ricaricato = torch.load(path_modello_ricaricato)
        model = classe_modello(dropout_rate=DROPOUT, nome_rete=NOME_RETE).to(device)
        model.load_state_dict(modello_ricaricato['model_state_dict'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(modello_ricaricato['optimizer_state_dict'])

        model.eval()
    
    risultati_nome_cartella = os.path.join(path_cartella_risultati, NOME_RETE)
    if not os.path.exists(risultati_nome_cartella):
        os.makedirs(risultati_nome_cartella)

    r2_scores= load_r2_scores(path_dir=risultati_nome_cartella, model_name=model.name)
    test_dir = os.path.join(BASE_DIR, 'dataset', dataset_scelto, 'test')

    #BEST PREDICTION
    colore_pred, colore_reale = visualizza_pred_reale(
        nome_rete = NOME_RETE,
        nome_img_app = nome_img,
        cartella_risultati = path_cartella_best_json,
        cartella_test = test_dir,
        mse=r2_scores["best_mse"], 
        rmse=r2_scores["best_rmse"],
        is_best=True
    )

    visualizza_metriche_immagini(
        nome_rete = NOME_RETE,
        nome_img_app = nome_img,
        cartella_risultati = path_cartella_best_json,
        cartella_test = test_dir,
        mse=r2_scores["best_mse"], 
        rmse=r2_scores["best_rmse"],
        is_best=True,
        colore_pred=colore_pred,
        colore_reale=colore_reale
    )
    #######

    #WORST PREDICTION
    colore_pred, colore_reale = visualizza_pred_reale(
        nome_rete = NOME_RETE,
        nome_img_app = nome_img_peggiore,
        cartella_risultati = path_cartella_best_json,
        cartella_test = test_dir,
        mse=r2_scores["best_mse"], 
        rmse=r2_scores["best_rmse"],
        is_best=False
    )

    visualizza_metriche_immagini(
        nome_rete = NOME_RETE,
        nome_img_app = nome_img_peggiore,
        cartella_risultati = path_cartella_best_json,
        cartella_test = test_dir,
        mse=r2_scores["best_mse"], 
        rmse=r2_scores["best_rmse"],
        is_best=False,
        colore_pred=colore_pred,
        colore_reale=colore_reale
    )
    #########

if __name__ == "__main__":
    #====================================================
    #              IPER-PARAMETRI E COSTANTI
    SEED = 42
    LEARNING_RATE = 0.0001
    LEARNING_RATE = 0.000005
    BATCH_SIZE = 4
    EPOCHS = 100
    PATIENCE = 10
    DROPOUT = 0.4
    SCHED_STEP= 20
    SCHED_GAMMA = 0.7

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #=====================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_scelto = "dataset_cilindro_spiralato"

    train_dataset = DsSingola_6Parametri(
    image_dir = 'dataset/' + dataset_scelto + '/train/images', 
    json_dir = 'dataset/' + dataset_scelto + '/train/labels', 
    apply_transforms=True
    )

    val_dataset = DsSingola_6Parametri(
        image_dir = 'dataset/' + dataset_scelto + '/val/images', 
        json_dir = 'dataset/' + dataset_scelto + '/val/labels', 
        apply_transforms=False
    )

    test_dataset = DsSingola_6Parametri(
        image_dir = 'dataset/' + dataset_scelto + '/test/images', 
        json_dir = 'dataset/' + dataset_scelto + '/test/labels', 
        apply_transforms=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2, shuffle=True, pin_memory=True)

    print(f"Train set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Test set size: {len(test_loader.dataset)}")


    #Useful for viewing multiple graphs of the same model
    """ risultati_modificato_path =os.path.join(BASE_DIR, 'results', 'CilindroSpiralato')
    percorso_log = [f"{risultati_modificato_path}/{MODELLI_DA_TESTARE[0]}_CilindroSpiralato/loss_log.json",
                    f"{risultati_modificato_path}/{MODELLI_DA_TESTARE[1]}_CilindroSpiralato/loss_log.json",
                    f"{risultati_modificato_path}/{MODELLI_DA_TESTARE[2]}_CilindroSpiralato/loss_log.json",
                    f"{risultati_modificato_path}/{MODELLI_DA_TESTARE[3]}_CilindroSpiralato/loss_log.json"]
   

    plot_modelli(MODELLI_DA_TESTARE, percorso_log)
    sys.exit() """
    
    #Classes related to the created hybrid models
    MODELLI_DA_TESTARE = [
        EFFNETB0_UNLOCK_DROP_EFFK_6,
        EFFNETB0_UNLOCK_DROP_FOURIER_6,
        SQUEEZENET_DROP4_FOURIER_DROP_256
    ]

    #Make sure to have rendered the images in Blender and that there is a match!
    IMG_MAP = {
        "EFFNETB0_UNLOCK_DROP_EFFK_6": {
            "best_image":   "MIGLIORE_3DfORMULA_179_vers2.png",
            "worst_image":  "PEGGIORE_3DfORMULA_11_vers0.png"
        },
        "EFFNETB0_UNLOCK_DROP_FOURIER_6": {
            "best_image":   "MIGLIORE_3DfORMULA_100_vers3.png",
            "worst_image":  "PEGGIORE_3DfORMULA_11_vers4.png"
        },
        "SQUEEZENET_DROP4_FOURIER_DROP_256": {
            "best_image":   "MIGLIORE_3DfORMULA_116_vers2.png",
            "worst_image":  "PEGGIORE_3DfORMULA_62_vers0.png"
        }
    }

    for classe_modello  in MODELLI_DA_TESTARE:
        nome = classe_modello .__name__

        # 1) Training and metrics calculation
        """ 
        print(f"\n\n--- INIZIO TRAINING: {nome} ---\n\n")
        esegui_training_con_modello(classe_modello)

        print(f"\n\n--- INIZIO TESTING: {nome} ---\n\n")
        esegui_test(classe_modello)
        """

        # 2) Results visualization
        #It’s essential to run the render in Blender first to have the image name!
        img_data = IMG_MAP.get(nome)

        if img_data is None:
            print(f"❌ Nessuna immagine associata a {nome}, salto.")
            continue
        
        visualizza_confronti_immagini(classe_modello , img_data["best_image"], img_data["worst_image"])
