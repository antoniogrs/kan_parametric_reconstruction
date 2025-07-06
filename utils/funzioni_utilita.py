import os
import json
import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib.cm as cm


from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial import cKDTree
from scipy.spatial.distance import directed_hausdorff
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm
from trimesh import Trimesh

# =============================================================================================================================================
# Creazione DATASET
#========================

class DsSingola_6Parametri(Dataset):
    '''
    Custom PyTorch Dataset for loading images and corresponding 6-parameter targets
    from JSON files, designed for training models on CilindroSpiralato dataset

    Key features:
        Loads image and JSON pairs from specified directories.
        Applies data augmentation transforms during training (resize, random flips, rotations, color jitter, normalization).
        Supports a basic transform pipeline without augmentation for validation/testing.
        Normalizes target parameters based on predefined min and max ranges for each parameter (radius, height, twist amplitude/frequency, wave amplitude/frequency)
        Returns the image filename, transformed image tensor, and normalized target tensor.

    Args:
        image_dir (str): Directory containing input images.
        json_dir (str): Directory containing JSON files with target parameters.
        apply_transforms (bool): Whether to apply data augmentation transforms (default: True).

    Methods:
        normalize_target: Normalizes target parameters to [0,1] range based on predefined min/max values.

    Returns:
        (from __getitem__) -> tuple: (image filename (str), transformed image tensor (torch.Tensor), normalized target tensor (torch.Tensor))
    '''
    def __init__(self, image_dir, json_dir, apply_transforms=True):
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.apply_transforms = apply_transforms

        # Trasformazioni
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
        ])

        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485]*3, std=[0.229]*3)
        ])

        self.data_pairs = []
        for json_file in os.listdir(self.json_dir):
            if not json_file.endswith(".json"):
                continue
            json_path = os.path.join(self.json_dir, json_file)
            with open(json_path, 'r') as f:
                data = json.load(f)
                image_filename = os.path.basename(data["image_file"])
                # Aggiungo la coppia mantenendo lo stesso ordine di caricamento
                self.data_pairs.append((image_filename, json_file))

        # Intervalli per la normalizzazione dei target
        self.target_min = {
            'radius': 0.5, 'height': 1.0,
            'twist_amp': 0.5, 'twist_freq': 1.0,
            'wave_amp': 0.0, 'wave_freq': 1.0
        }
        
        self.target_max = {
            'radius': 1.5, 'height': 3.0,
            'twist_amp': 2.0, 'twist_freq': 5.0,
            'wave_amp': 0.5, 'wave_freq': 4.0
        }

    def normalize_target(self, target):
        
        normalized = []
        keys = ['radius', 'height', 'twist_amp', 'twist_freq', 'wave_amp', 'wave_freq']
        for i, key in enumerate(keys):
            mn = self.target_min[key]
            mx = self.target_max[key]
            normalized.append((target[i] - mn) / (mx - mn))
        return torch.tensor(normalized, dtype=torch.float32)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        img_name, json_name = self.data_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.apply_transforms:
            image = self.train_transform(image)
        else:
            image = self.base_transform(image)

        json_path = os.path.join(self.json_dir, json_name)
        with open(json_path, 'r') as f:
            data = json.load(f)

        target = torch.tensor([
            data["radius"],
            data["height"],
            data["twist_amp"],
            data["twist_freq"],
            data["wave_amp"],
            data["wave_freq"]
        ], dtype=torch.float32)

        normalized_target = self.normalize_target(target)

        return img_name, image, normalized_target

def denormalize_target(normalized_target):
    target_min = {
        'radius': 0.5, 'height': 1.0, 'twist_amp': 0.5, 'twist_freq': 1.0, 'wave_amp': 0.0, 'wave_freq': 1.0
    }
    target_max = {
        'radius': 1.5, 'height': 3.0, 'twist_amp': 2.0, 'twist_freq': 5.0, 'wave_amp': 0.5, 'wave_freq': 4.0
    }

    denormalized = []
    for i, key in enumerate(['radius', 'height', 'twist_amp', 'twist_freq', 'wave_amp', 'wave_freq']):
        denormalized_value = normalized_target[i] * (target_max[key] - target_min[key]) + target_min[key]
        denormalized.append(denormalized_value)
    
    return denormalized

# ====================================================================================================================================================







# ====================================================================================================================================================
# Funzioni per TRAINING e TEST 
#==============================

def train(model, device, train_loader, val_loader, criterion, optimizer, epochs, patience=None, log_writer=None,
          path=None, path_json=None, model_name=None, scheduler_step_size=None, scheduler_gamma=None, previous_loss_data=None):
    '''
        Trains a PyTorch model with optional validation, early stopping, learning rate scheduling,
        logging, and checkpoint saving.

        Args:
            model (torch.nn.Module): the model to be trained
            device (torch.device): device to run the model on (CPU or GPU)
            train_loader (DataLoader): data loader for the training set
            val_loader (DataLoader): data loader for the validation set
            criterion (loss function): loss function to optimize
            epochs (int): number of training epochs
            patience (int, optional): number of epochs to wait before early stopping
            log_writer (LogWriter, optional): custom logger for plotting losses
            path (str, optional): directory to save trained model checkpoints
            path_json (str, optional): directory to save training/validation loss logs in JSON
            model_name (str, optional): name used for saving model and logs
            scheduler_step_size (int, optional): step size for StepLR scheduler
            scheduler_gamma (float, optional): decay factor for StepLR scheduler
            previous_loss_data (dict, optional): dictionary with previous 'train_loss', 'val_loss', and 'epochs'

        Returns:
            train_losses (list): training loss for each epoch
            val_losses (list): validation loss for each epoch
            all_epochs (list): list of completed epoch indices
    '''
    #Count of best model updates to be saved
    conta_miglior_modello = 0
    model.to(device)
    train_losses = previous_loss_data["train_loss"] if previous_loss_data else []
    val_losses = previous_loss_data["val_loss"] if previous_loss_data else []
    all_epochs = previous_loss_data["epochs"] if previous_loss_data else []

    model_path = os.path.join(path, model_name)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if path_json:
        risultati_nome_cartella = os.path.join(path_json, model_name)
        if not os.path.exists(risultati_nome_cartella):
            os.makedirs(risultati_nome_cartella) 

    if risultati_nome_cartella:
        loss_data = previous_loss_data if previous_loss_data else {
            "train_loss": [],
            "val_loss": [],
            "epochs": []
        }

    best_val_loss = float('inf') 
    epochs_without_improvement = 0

    scheduler = None
    if scheduler_step_size is not None and scheduler_gamma is not None:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    for epoch in range(epochs):
        model.train()
        all_epochs.append(epoch)
        epoch_train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training", ncols=100,
                         bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
        for image_filename, images, targets in train_bar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            train_bar.set_postfix(loss=f"{epoch_train_loss / len(train_loader):.4f}")

        model.eval()
        epoch_val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation", ncols=100,
                       bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}", leave=False)
        with torch.no_grad():
            for image_filename, images, targets in val_bar:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
                val_bar.set_postfix(val_loss=f"{epoch_val_loss / len(val_loader):.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if log_writer:
            log_writer.plot_train_loss(avg_train_loss, epoch+1)
            log_writer.plot_eval_loss(avg_val_loss, epoch+1)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

        # Update the scheduler (if active) and print the current learning rate
        if scheduler:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"⚡Learning Rate: {current_lr:.6f}")

        # Save logs to JSON file in the results directory
        if path_json:
            loss_data["train_loss"].append(avg_train_loss),
            loss_data["val_loss"].append(avg_val_loss),
            loss_data["epochs"].append(epoch)
            
            # Full path for the JSON file
            json_path = os.path.join(risultati_nome_cartella, "loss_log.json")
            with open(json_path, 'w') as f:
                json.dump(loss_data, f)

        # Early stopping e salvataggio del modello migliore
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            if path:
                conta_miglior_modello += 1
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, os.path.join(model_path, f"{model_name}_best.pt"))
                print(f"🙏 Nuovo Miglior Modello aggiornato: {conta_miglior_modello}")
        else:
            epochs_without_improvement += 1

        if patience:
            if epochs_without_improvement >= patience:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break

        # Salva il modello finale (al termine dell'addestramento o in seguito all'early stopping)
        if path:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(model_path, f"{model_name}.pt"))
    
    # Close log writer if it exists
    if log_writer:
        log_writer.close()

    return train_losses, val_losses, all_epochs

def plot_modelli(model_names, log_paths):
    '''
        Plots training and validation loss curves for 1 to 4 models for direct comparison.

        For each model, it loads the loss history (train and validation) from a JSON file
        and visualizes them using matplotlib. It assigns up to 8 distinct colors from the 'tab10'
        colormap (4 solid line for training curves, 4 dashed line with transparency for validation curves).

        Args:
            model_names (list of str): list of model names to label the curves in the plot
            log_paths (list of str): list of paths to the corresponding JSON log files (each containing
                                'train_loss', 'val_loss', and optionally 'epochs')
        Raises:
            ValueError: if number of models is not between 1 and 4, or if model_names and log_paths lengths mismatch
    '''
    n = len(model_names)
    if n == 0 or n > 4 or len(log_paths) != n:
        raise ValueError("Devi fornire tra 1 e 4 modelli, e log_paths deve avere la stessa lunghezza di model_names.")
    
    plt.figure(figsize=(12, 6))
    
    tab10 = cm.get_cmap('tab10').colors  # RGBA color
    base_colors = tab10[:n]
    
    # Create two color lists: solid for training, same colors with transparency for validation
    train_colors = base_colors
    val_colors   = [(r, g, b, 0.7) for (r, g, b) in base_colors] 

    for i, (name, path) in enumerate(zip(model_names, log_paths)):
        if not os.path.exists(path):
            print(f"❌ File non trovato per {name}: {path}")
            continue
        
        with open(path, 'r') as f:
            log = json.load(f)
        
        train_loss = log.get("train_loss", [])
        val_loss   = log.get("val_loss", [])
        epochs     = log.get("epochs", list(range(1, len(train_loss)+1)))
        
        plt.plot(epochs, train_loss, label=f"{name} Train",
                 color=train_colors[i], linestyle='-',
                 linewidth=2)
        plt.plot(epochs, val_loss,   label=f"{name} Val",
                 color=val_colors[i],   linestyle='--',
                 linewidth=2)
    
    plt.xlabel("Epoche")
    plt.ylabel("Loss")
    plt.title("Confronto Training e Validation Loss")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.7)
    plt.tight_layout()
    plt.show()

def calcola_r2_6Parametri(model, dataloader, device, set_name="Validation"):
    '''
        Computes the R² score between model predictions and targets over a dataset.
    '''
    print(f"📈 Avvio calcolo R² sul {set_name} set ...")
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for _, data, target in tqdm(dataloader, desc=f"Calcolo R² ({set_name})", unit="batch"):
            data, target = data.to(device), target.to(device)
            output = model(data)

            all_targets.append(target.cpu().numpy())
            all_predictions.append(output.cpu().numpy())

    # Concatenazione e calcolo finale
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    r2 = r2_score(all_targets, all_predictions)
    print(f"✅ R² Score sul {set_name} set: {r2:.4f}")
    return r2

def calcola_metriche(model, dataloader, device, set_name="Validation"):
    '''
        Computes multiple regression metrics (MAE, MSE, RMSE, R²) for a model on a given dataset.

        Args:
            model (torch.nn.Module): the trained model to evaluate
            dataloader (DataLoader): data loader providing input-target pairs
            device (torch.device): computation device (CPU or GPU)
            set_name (str, optional): name of the dataset split (e.g., "Validation" or "Test")

        Returns:
            dict: a dictionary containing MAE, MSE, RMSE, and R² score
    '''
    print(f"📈 Avvio calcolo metriche sul {set_name} set ...")
    model.eval()
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for _, data, target in tqdm(dataloader, desc=f"Calcolo metriche ({set_name})", unit="batch"):
            data, target = data.to(device), target.to(device)
            output = model(data)

            all_targets.append(target.cpu().numpy())
            all_predictions.append(output.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)

    mae = mean_absolute_error(all_targets, all_predictions)
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_predictions)

    print(f"✅ MAE sul {set_name} set: {mae:.4f}")
    print(f"✅ MSE sul {set_name} set: {mse:.4f}")
    print(f"✅ RMSE sul {set_name} set: {rmse:.4f}")
    print(f"✅ R² Score sul {set_name} set: {r2:.4f}")

    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

def convert_to_python_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_python_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    else:
        return obj

def stampa_best_prediction(best_actual_data, best_filename):
    best_prediction_json = {
        "radius": best_actual_data["radius"],
        "height": best_actual_data["height"],
        "twist_amp": best_actual_data["twist_amp"],
        "twist_freq": best_actual_data["twist_freq"],
        "wave_amp": best_actual_data["wave_amp"],
        "wave_freq": best_actual_data["wave_freq"],
        "camera_angles": best_actual_data["camera_angles"],
        "image_file": best_filename
    }

    print("--- Label Reali ---")
    for chiave, valore in best_prediction_json.items():
        print(f"{chiave}: {valore}") 

def best_pred_Test(model, test_loader, device, json_dir, save_dir, model_name):
    '''
        Evaluates a trained model on the test set and finds the image with the best prediction
        based on the lowest MSE (mean squared error).

        Args:
            model (torch.nn.Module): trained model to evaluate
            test_loader (DataLoader): DataLoader for the test dataset
            device (torch.device): computation device (CPU or GPU)
            json_dir (str): directory containing ground truth .json files
            save_dir (str): directory where the best prediction will be saved
            model_name (str): name of the model (used in output file naming)

        Returns:
            tuple: (best_mse, best_rmse) for the image with the most accurate prediction
    '''
    print(f"Avvio calcolo Test per il modello {model_name} ...")
    model.eval()
    best_mse = float("inf") 
    best_rmse = float("inf")
    best_pred = None
    best_filename = ""
    best_actual_data = None

    all_targets = []
    all_predictions = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for filenames, images, targets in test_loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()

            for i, filename in enumerate(filenames):
                json_file = os.path.join(json_dir, f"param_{filename.replace('.png', '.json')}")
                if not os.path.exists(json_file):
                    print(f"⚠️  File JSON mancante: {json_file}, saltato.")
                    continue

                with open(json_file, 'r') as f:
                    actual_data = json.load(f)

                actual_values = np.array([
                    actual_data["radius"], actual_data["height"], actual_data["twist_amp"],
                    actual_data["twist_freq"], actual_data["wave_amp"], actual_data["wave_freq"]
                ])
                pred_values = outputs[i]
                denormalized_pred_values = denormalize_target(pred_values)

                all_targets.append(actual_values)
                all_predictions.append(denormalized_pred_values)

                mse = mean_squared_error(actual_values, denormalized_pred_values)

                if mse < best_mse:
                    best_mse = mse
                    best_rmse = np.sqrt(best_mse)
                    best_pred = denormalized_pred_values
                    best_filename = filename
                    best_actual_data = actual_data

    best_prediction_json = {
        "radius": best_pred[0],
        "height": best_pred[1],
        "twist_amp": best_pred[2],
        "twist_freq": best_pred[3],
        "wave_amp": best_pred[4],
        "wave_freq": best_pred[5],
        "best_mse": best_mse,
        "best_rmse": best_rmse,
        "camera_angles": best_actual_data["camera_angles"],
        "image_file": best_filename,
    }

    result_file_path = os.path.join(save_dir, f"{model_name}_best_prediction_mse.json")

    with open(result_file_path, 'w') as f:
        json.dump(convert_to_python_serializable(best_prediction_json), f, indent=4)

    print(f"\n📊 Miglior immagine: {best_filename}")
    print(f"Best MSE: {best_mse:.5f} - Best RMSE: {best_rmse:.5f}")
    print(f"Previsione salvata in: {result_file_path}")
    stampa_best_prediction(best_actual_data, best_filename)
    return best_mse, best_rmse

def peggiore_pred_Test(model, test_loader, device, json_dir, save_dir, model_name):
    """
    Evaluates a trained model on the test set and finds the image with the worst prediction based on the highest MSE (mean squared error).

    Args:
        model (torch.nn.Module): trained model to evaluate
        test_loader (DataLoader): DataLoader for the test dataset
        device (torch.device): computation device (CPU or GPU)
        json_dir (str): directory containing ground truth .json files
        save_dir (str): directory where the worst prediction will be saved
        model_name (str): name of the model (used in output file naming)

    Returns:
        tuple: (worst_mse, worst_rmse) for the image with the largest prediction error
    """
    print(f"Avvio calcolo peggiore Test per il modello {model_name} ...")
    model.eval()
    worst_mse = -float("inf") 
    worst_rmse = -float("inf")
    worst_pred = None
    worst_filename = ""
    worst_actual_data = None

    all_targets = []
    all_predictions = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for filenames, images, targets in test_loader:
            images = images.to(device)
            outputs = model(images).cpu().numpy()

            for i, filename in enumerate(filenames):
                json_file = os.path.join(json_dir, f"param_{filename.replace('.png', '.json')}")
                if not os.path.exists(json_file):
                    print(f"⚠️  File JSON mancante: {json_file}, saltato.")
                    continue

                with open(json_file, 'r') as f:
                    actual_data = json.load(f)

                actual_values = np.array([
                    actual_data["radius"], actual_data["height"], actual_data["twist_amp"],
                    actual_data["twist_freq"], actual_data["wave_amp"], actual_data["wave_freq"]
                ])
                pred_values = outputs[i]
                denormalized_pred_values = denormalize_target(pred_values)

                all_targets.append(actual_values)
                all_predictions.append(denormalized_pred_values)

                mse = mean_squared_error(actual_values, denormalized_pred_values)

                if mse > worst_mse:
                    worst_mse = mse
                    worst_rmse = np.sqrt(worst_mse)
                    worst_pred = denormalized_pred_values
                    worst_filename = filename
                    worst_actual_data = actual_data

    worst_prediction_json = {
        "radius": worst_pred[0],
        "height": worst_pred[1],
        "twist_amp": worst_pred[2],
        "twist_freq": worst_pred[3],
        "wave_amp": worst_pred[4],
        "wave_freq": worst_pred[5],
        "worst_mse": worst_mse,
        "worst_rmse": worst_rmse,
        "camera_angles": worst_actual_data["camera_angles"],
        "image_file": worst_filename,
    }

    result_file_path = os.path.join(save_dir, f"{model_name}_worst_prediction_mse.json")

    with open(result_file_path, 'w') as f:
        json.dump(convert_to_python_serializable(worst_prediction_json), f, indent=4)

    print(f"\n📊 Peggior immagine: {worst_filename}")
    print(f"Worst MSE: {worst_mse:.5f} - Worst RMSE: {worst_rmse:.5f}")
    print(f"Previsione salvata in: {result_file_path}")
    stampa_best_prediction(worst_actual_data, worst_filename)
    return worst_mse, worst_rmse

def genera_mesh_da_parametri(parametri, u_steps=200, v_steps=200):
    '''
        Generates a 3D mesh (vertices and faces) of a spiral-shaped cylinder based on parametric inputs.

        Args:
            parametri (dict): dictionary containing the shape parameters:
                'radius' (float): base radius of the cylinder
                'height' (float): height of the cylinder
                'twist_amp' (float): amplitude of the spiral twist
                'twist_freq' (float): frequency of the twist modulation
                'wave_amp' (float): amplitude of the radial wave
                'wave_freq' (float): frequency of the wave modulation
            u_steps (int): number of subdivisions around the cylinder's circumference
            v_steps (int): number of subdivisions along the cylinder's height

        Returns:
            vertices (np.ndarray): array of 3D points defining the mesh surface
            faces (np.ndarray): array of face indices (quads) for mesh construction
    '''
    radius = parametri['radius']
    height = parametri['height']
    twist_amp = parametri['twist_amp']
    twist_freq = parametri['twist_freq']
    wave_amp = parametri['wave_amp']
    wave_freq = parametri['wave_freq']

    vertices = []
    faces = []

    # Generazione vertici
    for i in range(u_steps + 1):
        u = i * 2 * np.pi / u_steps
        for j in range(v_steps + 1):
            v = j * height / v_steps
            angle = u + twist_amp * math.sin(twist_freq * v)
            r_mod = radius + wave_amp * math.cos(wave_freq * v)
            x = r_mod * math.cos(angle)
            y = r_mod * math.sin(angle)
            z = v
            vertices.append((x, y, z))

    # Generazione facce
    for i in range(u_steps):
        for j in range(v_steps):
            a = i * (v_steps + 1) + j
            b = a + 1
            c = a + (v_steps + 1) + 1
            d = a + (v_steps + 1)
            faces.append([a, b, c, d])

    return np.array(vertices), np.array(faces)

def estrai_colore_oggetto(img, colore_sfondo=(198, 198, 198), soglia_grigio=20):
    img_np = np.array(img)
    diff = np.abs(img_np - colore_sfondo)
    mask = np.any(diff > soglia_grigio, axis=-1)
    if np.sum(mask) == 0:
        return (255, 165, 0)
    oggetto_pixels = img_np[mask]
    colore_medio = np.mean(oggetto_pixels, axis=0)
    return tuple(colore_medio.astype(int))

def draw_mesh(ax, verts, faces, color, alpha=0.7, linewidths=None):
    '''
        Draws a 3D mesh on the given matplotlib 3D axis with optional transparency and edge line width.

        Args:
            ax (Axes3D): the 3D axis to draw the mesh on
            verts (np.ndarray): array of 3D vertex coordinates (N x 3)
            faces (list or np.ndarray): list of face indices defining each polygon (typically quads)
            color (str or tuple): face and edge color of the mesh
            alpha (float, optional): transparency of the mesh (default is 0.7)
            linewidths (float, optional): edge line width (default is 0.2 if not specified)

        The function also disables axis ticks for a cleaner visualization.
    '''
    
    lw = linewidths if linewidths is not None else 0.2

    poly = Poly3DCollection([verts[f] for f in faces],
                            facecolors=color, linewidths=lw,
                            edgecolors=color, alpha=alpha)

    ax.add_collection3d(poly)
    ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

def visualizza_pred_reale(
    nome_rete, nome_img_app, cartella_risultati, cartella_test, struttura_modello=None,
    mse=None, rmse=None, is_best=True
):
    '''
        Visualizes and compares the predicted and real 3D object images and meshes side by side.

        This function:
        - Loads prediction metadata (best or worst) from JSON logs based on the 'is_best' flag.
        - Loads and resizes the predicted and real images to match dimensions.
        - Extracts relevant geometric parameters and computes differences.
        - Generates 3D mesh representations for both predicted and real objects.
        - Extracts dominant colors from images to color the meshes consistently.
        
        Creates a 2x2 matplotlib figure with:
            1) Real image
            2) Predicted image,
            3) 3D mesh of the real object with transparency,
            4) 3D mesh of the predicted object with transparency.

        Args:
            nome_rete (str): name of the model/network used for prediction.
            nome_img_app (str): filename of the predicted image.
            cartella_risultati (str): directory containing prediction JSON and images.
            cartella_test (str): directory containing ground truth test images and labels.
            struttura_modello (optional): model structure (not used here, reserved for future).
            mse (float, optional): mean squared error value (unused in visualization).
            rmse (float, optional): root mean squared error value (unused in visualization).
            is_best (bool): flag to select best or worst prediction JSON log (default True).

        Returns:
            tuple of two RGB color arrays: (color_predicted, color_real), extracted from the respective images.
    '''
    if is_best:
        json_file = os.path.join(cartella_risultati, f"{nome_rete}_best_prediction_mse.json")
    else:
        json_file = os.path.join(cartella_risultati, f"{nome_rete}_peggior_mse.json")
    with open(json_file, 'r') as f:
        dati_pred = json.load(f)

    img_app_path = os.path.join(cartella_risultati, nome_img_app)
    nome_img_reale = dati_pred.get("image_file")
    img_reale_path = os.path.join(cartella_test, 'images', nome_img_reale)
    label_reale_path = os.path.join(cartella_test, 'labels', f"param_{os.path.splitext(nome_img_reale)[0]}.json")

    img_app = Image.open(img_app_path).convert('RGB')
    img_reale = Image.open(img_reale_path).convert('RGB')
    if img_app.size != img_reale.size:
        img_reale = img_reale.resize(img_app.size)

    parametri = ['radius', 'height', 'twist_amp', 'twist_freq', 'wave_amp', 'wave_freq']
    valori_pred = dati_pred.get("prediction", dati_pred)
    with open(label_reale_path, 'r') as f:
        dati_reali = json.load(f)
    valori_reali = {k: dati_reali.get(k, 0.0) for k in parametri}
    differenze = {k: valori_reali[k] - valori_pred.get(k, 0.0) for k in parametri}

    verts_pred, faces_pred = genera_mesh_da_parametri(valori_pred)
    verts_real, faces_real = genera_mesh_da_parametri(valori_reali)
    colore_pred = estrai_colore_oggetto(img_app)
    colore_reale = estrai_colore_oggetto(img_reale)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7.5))
    fig.suptitle("Differenze degli oggetti", fontsize=16, fontweight='bold', y=0.95)

    # 1) Immagine Reale
    ax1 = axs[0, 0]
    ax1.imshow(img_reale)
    ax1.axis('off')
    ax1.set_title("Immagine Reale", fontsize=12)

    # 2) Immagine Predetta
    ax3 = axs[0, 1]
    ax3.imshow(img_app)
    ax3.axis('off')
    ax3.set_title("Immagine Predetta", fontsize=12)

    # 3) Mesh Reale (sostituisce axs[1,0] con subplot 3D)
    fig.delaxes(axs[1, 0])
    ax2 = fig.add_subplot(2, 2, 3, projection='3d')
    draw_mesh(ax2, verts_real, faces_real, np.array(colore_reale) / 255.0, alpha=0.4)
    ax2.text2D(0.5, -0.1, "Mesh Reale", transform=ax2.transAxes,
            fontsize=12, ha="center", va="top")
    ax2.axis('off')

    # 4) Mesh Predetta (sostituisce axs[1,1] con subplot 3D)
    fig.delaxes(axs[1, 1])
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    draw_mesh(ax4, verts_pred, faces_pred, np.array(colore_pred) / 255.0, alpha=0.4)
    ax4.text2D(0.5, -0.1, "Mesh Predetta", transform=ax4.transAxes,
            fontsize=12, ha="center", va="top")
    ax4.axis('off')

    plt.subplots_adjust(
        left=0.05, right=0.95,
        bottom=0.05, top=0.90,
        wspace=0.03, hspace=0.08
    )

    etichetta = "BEST" if is_best else "PEGGIORE"
    #plt.savefig(f"C:/user_tex/IMMAGINI_RISULTATI/{nome_rete}_{etichetta}_CONFRONTO.png", dpi=1000, bbox_inches='tight', pad_inches=0.1)
    #plt.close()

    plt.show()
    
    return colore_pred, colore_reale

def visualizza_metriche_immagini(
    nome_rete, nome_img_app, cartella_risultati, cartella_test, struttura_modello=None,
    mse=None, rmse=None, is_best=True, colore_pred=None, colore_reale=None 
):
    '''
    Visualizes and compares the predicted and real 3D shapes and their corresponding 2D images,
    highlighting pixel-wise and geometric differences through overlay maps, RGB deltas, and Hausdorff metrics.

    This function loads prediction metadata (either best or worst depending on `is_best` flag),
    retrieves the associated images and mesh parameters, computes both visual (RGB) and geometric
    differences, and displays a 2x2 subplot comparison:
        1) Heatmap overlay between predicted and real image.
        2) RGB difference image.
        3) Real and predicted 3D meshes superimposed.
        4) Predicted mesh colored by vertex-wise Hausdorff distance.

    It also computes:
    - Symmetric Hausdorff distance between the predicted and real mesh.
    - Mean distance between vertices (predicted to real).
    - A color mapping (colormap 'jet') of vertex errors on the predicted mesh.
    - A colorbar legend for interpretability of geometric error.

    Args:
        nome_rete (str): Name of the model/network (used for file loading and title).
        nome_img_app (str): Filename of the approximated/predicted image.
        cartella_risultati (str): Path to the folder containing predictions and logs.
        cartella_test (str): Path to the folder containing real test images and labels.
        struttura_modello (optional): Placeholder for model structure (not used directly).
        mse (float, optional): Mean squared error for display (currently unused).
        rmse (float, optional): Root mean squared error for display (currently unused).
        is_best (bool): If True, uses best prediction JSON, else uses worst. Default is True.
        colore_pred (tuple or list): RGB color of the predicted image object.
        colore_reale (tuple or list): RGB color of the real image object.

    Raises:
        FileNotFoundError: If any of the required files (images or JSONs) are missing.

    Returns:
        None. The function displays a matplotlib 2x2 visualization plot containing:
            - Overlay Heatmap (2D image difference)
            - RGB Delta
            - 3D Mesh Comparison
            - 3D Mesh Error Heatmap (with Hausdorff distance)
    '''
    
    # 1) Selezione file JSON
    if is_best:
        json_file = os.path.join(cartella_risultati, f"{nome_rete}_best_prediction_mse.json")
    else:
        json_file = os.path.join(cartella_risultati, f"{nome_rete}_peggior_mse.json")

    if not os.path.exists(json_file):
        print(f"File JSON non trovato: {json_file}")
        return
    with open(json_file, 'r') as f:
        dati_pred = json.load(f)

    # 2) Caricamento immagini
    img_app_path = os.path.join(cartella_risultati, nome_img_app)
    if not os.path.exists(img_app_path):
        print(f"Immagine approssimata non trovata: {img_app_path}")
        return
    img_app = Image.open(img_app_path).convert('RGB')

    nome_img_reale = dati_pred.get("image_file")
    if not nome_img_reale:
        print(f"Nome immagine reale non trovato nel JSON: {json_file}")
        return
    img_reale_path = os.path.join(cartella_test, 'images', nome_img_reale)
    label_reale_path = os.path.join(
        cartella_test, 'labels', f"param_{os.path.splitext(nome_img_reale)[0]}.json"
    )
    if not os.path.exists(img_reale_path) or not os.path.exists(label_reale_path):
        print(f"Immagine o label reale non trovata: {img_reale_path}, {label_reale_path}")
        return
    img_reale = Image.open(img_reale_path).convert('RGB')
    if img_app.size != img_reale.size:
        img_reale = img_reale.resize(img_app.size)

    # 3) Calcolo differenze 2D e heatmap
    arr_app = np.array(img_app, dtype=float) / 255.0
    arr_real = np.array(img_reale, dtype=float) / 255.0
    diff_rgb = np.abs(arr_app - arr_real)
    diff_gray = np.linalg.norm(arr_app - arr_real, axis=2)
    ptp = np.ptp(diff_gray)
    diff_gray_norm = (diff_gray - diff_gray.min()) / (ptp + 1e-8)
    heatmap = cm.get_cmap('hot')(diff_gray_norm)[:, :, :3]
    overlay_arr = 0.6 * arr_real + 0.4 * heatmap
    img_overlay = Image.fromarray((overlay_arr * 255).astype('uint8'))

    # 4) Preparazione mesh
    with open(label_reale_path, 'r') as f:
        dati_reali = json.load(f)
    parametri = ['radius', 'height', 'twist_amp', 'twist_freq', 'wave_amp', 'wave_freq']
    valori_pred = dati_pred.get("prediction", dati_pred)
    if not isinstance(valori_pred, dict):
        valori_pred = dati_pred
    valori_reali = {k: dati_reali.get(k, 0.0) for k in parametri}

    verts_pred, faces_pred = genera_mesh_da_parametri(valori_pred)
    verts_real, faces_real = genera_mesh_da_parametri(valori_reali)
    verts_pred = np.array(verts_pred)
    verts_real = np.array(verts_real)

    # 5) Calcolo della distanza di Hausdorff simmetrica e colorazione
    # 5.1 Direzionale: predetto -> reale
    d1, _, _ = directed_hausdorff(verts_pred, verts_real)
    # 5.2 Direzionale: reale -> predetto
    d2, _, _ = directed_hausdorff(verts_real, verts_pred)
    # 5.3 Distanza di Hausdorff simmetrica
    hausdorff_dist = max(d1, d2)

    # 5.4 Calcolo delle distanze minime pred->real per ogni vertice predetto
    #     (necessario per color mapping)
    tree_real = cKDTree(verts_real)
    dists, _ = tree_real.query(verts_pred, k=1)

    # 5.5 Normalizzazione e colormap
    d_norm = (dists - dists.min()) / (np.ptp(dists) + 1e-12)
    vert_colors = cm.get_cmap('jet')(d_norm)[:, :3]

    # 5.6 Colore medio per faccia
    face_colors = [vert_colors[f].mean(axis=0) for f in faces_pred]# 6) Plot 2x2
     # 6) Plot 2x2 with smaller figure size
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))  # Reduced from (16, 12) to (12, 9)
    # 1) titolo generale centrato
    fig.suptitle("Confronto 2D e 3D", fontsize=16, fontweight='bold', ha='center')

    # Plot overlay
    axs[0, 0].imshow(img_overlay); axs[0, 0].axis('off'); axs[0, 0].set_title("Overlay Heatmap", fontsize=12)  # Reduced fontsize
    # Plot diff RGB
    axs[0, 1].imshow((diff_rgb * 255).astype('uint8')); axs[0, 1].axis('off'); axs[0, 1].set_title("Differenza RGB", fontsize=12)  # Reduced fontsize

    # Mesh reale+predetta
    fig.delaxes(axs[1, 0])
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    # Disegna bordi per evidenziare la sovrapposizione
    draw_mesh(ax3, verts_real, faces_real, color=np.array(colore_reale)/255.0, alpha=0.1, linewidths=0.1)
    draw_mesh(ax3, verts_pred, faces_pred, color=np.array(colore_pred)/255.0, alpha=0.2, linewidths=0.2)
    ax3.text2D(0.5, -0.07, f"Distanza Media trai i Vertici: {dists.mean():.3f}", transform=ax3.transAxes,
               fontsize=14, ha="center", va="top")  # Reduced fontsize
    ax3.axis('off')
    ax3.set_title("Mesh Reale e Predetta", fontsize=14)  # Reduced fontsize

    # Mesh predetta colorata colonna 4
    fig.delaxes(axs[1, 1])
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    mesh = Poly3DCollection(verts_pred[faces_pred], facecolors=face_colors,
                            edgecolors='k', linewidths=0.05)
    ax4.add_collection3d(mesh)
    mn, mx = verts_pred.min(axis=0), verts_pred.max(axis=0)
    ax4.set_xlim(mn[0], mx[0]); ax4.set_ylim(mn[1], mx[1]); ax4.set_zlim(mn[2], mx[2])
    ax4.set_title(f"Distanza di Hausdorff Simmetrica", fontsize=12)  # Reduced fontsize
    ax4.text2D(0.5, -0.07,
            f"Valore massimo: {hausdorff_dist:.4f}",
            transform=ax4.transAxes,
            fontsize=12, ha="center", va="top")
    ax4.axis('off')
    
    # Aggiunta colorbar laterale per la colormap 'jet'
    norm = plt.Normalize(vmin=dists.min(), vmax=dists.max())
    sm = cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax4, fraction=0.03, pad=0.1, label='Distanza vertice predetto → reale')

    # Adjust layout with tighter margins
    plt.subplots_adjust(left=0.05, right=0.92, bottom=0.05, top=0.90, wspace=0.15, hspace=0.25)

    etichetta = "BEST" if is_best else "PEGGIORE"
    #plt.savefig(f"C:/user_tex/IMMAGINI_RISULTATI/{nome_rete}_{etichetta}_METRICHE.png", dpi=1000, bbox_inches='tight', pad_inches=0.1)
    plt.show()

def save_r2_scores(r2_scores: dict, path_dir: str, model_name: str):
    os.makedirs(path_dir, exist_ok=True)
    file_path = os.path.join(path_dir, f"{model_name}_r2_scores.json")
    with open(file_path, 'w') as f:
        json.dump(r2_scores, f, indent=4)
    print(f"✅ R² salvati in: {file_path}")

def load_r2_scores(path_dir: str, model_name: str):
    file_path = os.path.join(path_dir, f"{model_name}_r2_scores.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File R² non trovato in {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)
    
def load_best_pred(path_dir: str, model_name: str):
    file_path = os.path.join(path_dir, f"{model_name}_best_prediction_mse.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File best prediction non trovato in {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def save_metriche(r2_scores: dict, path_dir: str, model_name: str, nome_str:str):
    os.makedirs(path_dir, exist_ok=True)
    file_path = os.path.join(path_dir, f"{model_name}_metriche_{nome_str}.json")
    with open(file_path, 'w') as f:
        json.dump(r2_scores, f, indent=4)
    print(f"✅ metriche salvate in: {file_path}")

def load_metriche(path_dir: str, model_name: str, nome_str:str):
    file_path = os.path.join(path_dir, f"{model_name}_metriche_{nome_str}.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ File metriche non trevato in {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)