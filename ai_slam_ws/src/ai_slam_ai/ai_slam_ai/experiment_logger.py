"""
Moduł do logowania metadanych eksperymentów AI SLAM.

Zapisuje wszystkie dane związane z eksperymentem:
- Czas wykonania poszczególnych etapów
- Parametry konfiguracyjne
- Metryki wydajności
- Informacje o środowisku
- Ścieżki do artefaktów
"""

import json
import os
import platform
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field, asdict


def get_system_info() -> Dict[str, Any]:
    """Pobiera informacje o systemie."""
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }
    
    # Informacje o PyTorch (jeśli dostępne)
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
    except ImportError:
        pass
    
    # Informacje o pamięci (jeśli dostępne)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info["total_memory_gb"] = round(mem.total / (1024**3), 2)
        info["available_memory_gb"] = round(mem.available / (1024**3), 2)
        info["cpu_count"] = psutil.cpu_count()
    except ImportError:
        pass
    
    return info


@dataclass
class TimingInfo:
    """Informacje o czasie wykonania."""
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    def start(self):
        """Rozpoczyna pomiar czasu tylko jeśli nie był już rozpoczęty."""
        if self.start_time is None:  # Nie nadpisuj istniejącego czasu
            self.start_time = datetime.now().isoformat()
            self._start_ts = time.time()
    
    def end(self):
        """Kończy pomiar czasu tylko jeśli nie był już zakończony."""
        if self.end_time is None:  # Nie nadpisuj istniejącego czasu
            self.end_time = datetime.now().isoformat()
            if hasattr(self, '_start_ts'):
                self.duration_seconds = time.time() - self._start_ts
    
    def is_completed(self) -> bool:
        """Sprawdza czy pomiar został zakończony."""
        return self.end_time is not None and self.duration_seconds is not None


@dataclass
class DatasetMetadata:
    """Metadane zbierania datasetu."""
    timing: TimingInfo = field(default_factory=TimingInfo)
    parameters: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None
    file_size_mb: Optional[float] = None
    
    def set_parameters(self, seed: int, duration_sec: float, max_samples: int,
                      scan_topic: str, odom_topic: str, gt_topic: str):
        self.parameters = {
            "seed": seed,
            "target_duration_sec": duration_sec,
            "max_samples": max_samples,
            "scan_topic": scan_topic,
            "odom_topic": odom_topic,
            "gt_topic": gt_topic,
        }
    
    def set_statistics(self, n_samples: int, scan_dim: int, 
                       actual_duration_sec: float, samples_per_second: float):
        self.statistics = {
            "n_samples": n_samples,
            "scan_dimension": scan_dim,
            "actual_duration_sec": actual_duration_sec,
            "samples_per_second": samples_per_second,
        }


@dataclass
class TrainingMetadata:
    """Metadane treningu modelu."""
    timing: TimingInfo = field(default_factory=TimingInfo)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    training_results: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    history_path: Optional[str] = None
    
    def set_parameters(self, seed: int, max_epochs: int, patience: int,
                      min_delta: float, lr: float, val_ratio: float, batch_size: int):
        self.parameters = {
            "seed": seed,
            "max_epochs": max_epochs,
            "patience": patience,
            "min_delta": min_delta,
            "learning_rate": lr,
            "validation_ratio": val_ratio,
            "batch_size": batch_size,
        }
    
    def set_dataset_info(self, n_total: int, n_train: int, n_val: int, 
                         input_dim: int, output_dim: int):
        self.dataset_info = {
            "n_total_samples": n_total,
            "n_train_samples": n_train,
            "n_validation_samples": n_val,
            "input_dimension": input_dim,
            "output_dimension": output_dim,
        }
    
    def set_model_info(self, architecture: str, total_params: int, trainable_params: int):
        self.model_info = {
            "architecture": architecture,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }
    
    def set_training_results(self, epochs_run: int, best_epoch: int, 
                             best_val_loss: float, final_train_loss: float,
                             early_stopped: bool):
        self.training_results = {
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "best_validation_loss": best_val_loss,
            "final_training_loss": final_train_loss,
            "early_stopped": early_stopped,
        }


@dataclass
class InferenceMetadata:
    """Metadane inferencji."""
    timing: TimingInfo = field(default_factory=TimingInfo)
    parameters: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    model_path: Optional[str] = None
    
    def set_parameters(self, seed: int, scan_topic: str, odom_topic: str,
                      pose_topic: str, tf_parent: str, tf_child: str):
        self.parameters = {
            "seed": seed,
            "scan_topic": scan_topic,
            "odom_topic": odom_topic,
            "pose_topic": pose_topic,
            "tf_parent_frame": tf_parent,
            "tf_child_frame": tf_child,
        }
    
    def set_statistics(self, n_predictions: int, total_duration_sec: float,
                       avg_inference_time_ms: float):
        self.statistics = {
            "n_predictions": n_predictions,
            "total_duration_sec": total_duration_sec,
            "avg_inference_time_ms": avg_inference_time_ms,
            "predictions_per_second": n_predictions / total_duration_sec if total_duration_sec > 0 else 0,
        }


@dataclass
class EvaluationMetadata:
    """Metadane ewaluacji."""
    timing: TimingInfo = field(default_factory=TimingInfo)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    
    def set_parameters(self, seed: int, mode: str, duration_sec: float,
                      reference_map_yaml: str):
        self.parameters = {
            "seed": seed,
            "mode": mode,
            "duration_sec": duration_sec,
            "reference_map_yaml": reference_map_yaml,
        }
    
    def set_metrics(self, rmse_xy_baseline: float, rmse_theta_baseline: float,
                   rmse_xy_ai: Optional[float], rmse_theta_ai: Optional[float],
                   iou_map_baseline: Optional[float], iou_map_ai: Optional[float],
                   n_samples: int, iou_map_robak: Optional[float] = None,
                   iou_map_rywak: Optional[float] = None):
        self.metrics = {
            "rmse_xy_baseline": rmse_xy_baseline,
            "rmse_theta_baseline": rmse_theta_baseline,
            "iou_map_baseline": iou_map_baseline,
            "rmse_xy_ai": rmse_xy_ai,
            "rmse_theta_ai": rmse_theta_ai,
            "iou_map_ai": iou_map_ai,
            "iou_map_robak": iou_map_robak,
            "iou_map_rywak": iou_map_rywak,
            "n_evaluation_samples": n_samples,
        }
        
        # Oblicz poprawę AI względem baseline
        if rmse_xy_ai is not None and rmse_xy_baseline > 0:
            self.metrics["rmse_xy_improvement_percent"] = \
                round((rmse_xy_baseline - rmse_xy_ai) / rmse_xy_baseline * 100, 2)
        if rmse_theta_ai is not None and rmse_theta_baseline > 0:
            self.metrics["rmse_theta_improvement_percent"] = \
                round((rmse_theta_baseline - rmse_theta_ai) / rmse_theta_baseline * 100, 2)
        if iou_map_ai is not None and iou_map_baseline is not None and iou_map_baseline > 0:
            self.metrics["iou_improvement_percent"] = \
                round((iou_map_ai - iou_map_baseline) / iou_map_baseline * 100, 2)


@dataclass
class ExperimentLog:
    """Główny log eksperymentu zawierający wszystkie etapy."""
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    system_info: Dict[str, Any] = field(default_factory=get_system_info)
    
    dataset: DatasetMetadata = field(default_factory=DatasetMetadata)
    training: TrainingMetadata = field(default_factory=TrainingMetadata)
    inference: InferenceMetadata = field(default_factory=InferenceMetadata)
    evaluation: EvaluationMetadata = field(default_factory=EvaluationMetadata)
    
    total_experiment_time_sec: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje log do słownika."""
        return {
            "experiment_id": self.experiment_id,
            "created_at": self.created_at,
            "system_info": self.system_info,
            "dataset": asdict(self.dataset),
            "training": asdict(self.training),
            "inference": asdict(self.inference),
            "evaluation": asdict(self.evaluation),
            "total_experiment_time_sec": self.total_experiment_time_sec,
            "notes": self.notes,
        }
    
    def add_note(self, note: str):
        """Dodaje notatkę do logu."""
        timestamp = datetime.now().isoformat()
        self.notes.append(f"[{timestamp}] {note}")


def generate_experiment_id() -> str:
    """Generuje unikalny identyfikator eksperymentu z timestampem (jako string z prefiksem)."""
    return "exp_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def get_experiment_dir(base_out_dir: str, experiment_id: str) -> str:
    """
    Zwraca ścieżkę do katalogu eksperymentu.
    Struktura: base_out_dir/experiment_id/
    Jeśli experiment_id nie zaczyna się od 'exp_', dodaje ten prefiks.
    """
    # Upewnij się że experiment_id ma prefiks 'exp_' dla spójności
    if not experiment_id.startswith("exp_"):
        exp_folder = f"exp_{experiment_id}"
    else:
        exp_folder = experiment_id
    return os.path.join(os.path.abspath(base_out_dir), exp_folder)


class ExperimentLogger:
    """
    Centralny logger eksperymentów.
    
    Wszystkie dane eksperymentu są zapisywane w dedykowanym podfolderze:
    out_dir/exp_YYYYMMDD_HHMMSS/
    
    Używanie:
        # Generowanie ID eksperymentu (raz na cały eksperyment)
        experiment_id = generate_experiment_id()
        
        # W każdym node:
        logger = ExperimentLogger("/path/to/out_dir", experiment_id)
        
        # Przed zbieraniem datasetu
        logger.start_dataset_collection(seed=123, duration_sec=60, ...)
        
        # Po zakończeniu
        logger.end_dataset_collection(n_samples=1000, ...)
        
        # Na końcu eksperymentu
        logger.save()
    """
    
    METADATA_FILENAME = "experiment_metadata.json"
    
    def __init__(self, base_out_dir: str, experiment_id: Optional[str] = None, create_subdir: bool = True):
        """
        Inicjalizuje logger eksperymentu.
        
        Args:
            base_out_dir: Bazowy katalog wyjściowy (np. "out")
            experiment_id: Unikalny ID eksperymentu (jeśli None, zostanie wygenerowany)
            create_subdir: Czy tworzyć podfolder dla eksperymentu (domyślnie True)
        """
        self.base_out_dir = os.path.abspath(base_out_dir)
        
        # Jeśli experiment_id nie podano, spróbuj wczytać z istniejącego pliku lub wygeneruj nowy
        if experiment_id is None:
            # Szukaj ostatniego eksperymentu w base_out_dir
            experiment_id = self._find_or_create_experiment_id()
        
        self.experiment_id = experiment_id
        
        if create_subdir:
            self.out_dir = get_experiment_dir(self.base_out_dir, self.experiment_id)
        else:
            self.out_dir = self.base_out_dir
            
        os.makedirs(self.out_dir, exist_ok=True)
        
        self.metadata_path = os.path.join(self.out_dir, self.METADATA_FILENAME)
        
        # Sprawdź czy istnieje poprzedni log w tym katalogu
        if os.path.exists(self.metadata_path):
            self.log = self._load_existing()
        else:
            self.log = ExperimentLog(experiment_id=self.experiment_id)
        
        try:
            self._experiment_start = datetime.fromisoformat(self.log.created_at).timestamp()
        except Exception:
            self._experiment_start = time.time()
    
    def _find_or_create_experiment_id(self) -> str:
        """
        Szuka aktywnego eksperymentu w katalogu bazowym lub tworzy nowy ID.
        Aktywny eksperyment to taki, który ma plik metadata ale nie jest sfinalicowany.
        """
        if not os.path.isdir(self.base_out_dir):
            return generate_experiment_id()
        
        # Szukaj folderów exp_* posortowanych od najnowszego
        exp_folders = []
        for name in os.listdir(self.base_out_dir):
            if name.startswith("exp_") and os.path.isdir(os.path.join(self.base_out_dir, name)):
                exp_folders.append(name)
        
        exp_folders.sort(reverse=True)  # Najnowsze pierwsze
        
        for folder in exp_folders:
            metadata_path = os.path.join(self.base_out_dir, folder, self.METADATA_FILENAME)
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    # Sprawdź czy eksperyment nie jest zakończony
                    if data.get("total_experiment_time_sec") is None:
                        # Ten eksperyment jest aktywny
                        return data.get("experiment_id", folder.replace("exp_", ""))
                except Exception:
                    continue
        
        # Nie znaleziono aktywnego eksperymentu - utwórz nowy
        return generate_experiment_id()
    
    def get_output_dir(self) -> str:
        """Zwraca ścieżkę do katalogu wyjściowego eksperymentu."""
        return self.out_dir
    
    def _load_existing(self) -> ExperimentLog:
        """Wczytuje istniejący log (do kontynuacji eksperymentu)."""
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            log = ExperimentLog(
                experiment_id=data.get("experiment_id", "unknown"),
                created_at=data.get("created_at", datetime.now().isoformat()),
            )
            log.system_info = data.get("system_info", get_system_info())
            log.notes = data.get("notes", [])
            
            # Rekonstruuj metadane etapów
            if "dataset" in data:
                d = data["dataset"]
                log.dataset.parameters = d.get("parameters", {})
                log.dataset.statistics = d.get("statistics", {})
                log.dataset.file_path = d.get("file_path")
                log.dataset.file_size_mb = d.get("file_size_mb")
                if "timing" in d:
                    log.dataset.timing.start_time = d["timing"].get("start_time")
                    log.dataset.timing.end_time = d["timing"].get("end_time")
                    log.dataset.timing.duration_seconds = d["timing"].get("duration_seconds")
                    # Ustaw _start_ts na sztuczną wartość jeśli timing jest zakończony
                    if log.dataset.timing.duration_seconds is not None:
                        log.dataset.timing._start_ts = time.time() - log.dataset.timing.duration_seconds
            
            if "training" in data:
                t = data["training"]
                log.training.parameters = t.get("parameters", {})
                log.training.dataset_info = t.get("dataset_info", {})
                log.training.model_info = t.get("model_info", {})
                log.training.training_results = t.get("training_results", {})
                log.training.model_path = t.get("model_path")
                log.training.history_path = t.get("history_path")
                if "timing" in t:
                    log.training.timing.start_time = t["timing"].get("start_time")
                    log.training.timing.end_time = t["timing"].get("end_time")
                    log.training.timing.duration_seconds = t["timing"].get("duration_seconds")
                    # Ustaw _start_ts na sztuczną wartość jeśli timing jest zakończony
                    if log.training.timing.duration_seconds is not None:
                        log.training.timing._start_ts = time.time() - log.training.timing.duration_seconds
            
            if "inference" in data:
                i = data["inference"]
                log.inference.parameters = i.get("parameters", {})
                log.inference.statistics = i.get("statistics", {})
                log.inference.model_path = i.get("model_path")
                if "timing" in i:
                    log.inference.timing.start_time = i["timing"].get("start_time")
                    log.inference.timing.end_time = i["timing"].get("end_time")
                    log.inference.timing.duration_seconds = i["timing"].get("duration_seconds")
                    # Ustaw _start_ts na sztuczną wartość jeśli timing jest zakończony
                    if log.inference.timing.duration_seconds is not None:
                        log.inference.timing._start_ts = time.time() - log.inference.timing.duration_seconds
            
            if "evaluation" in data:
                e = data["evaluation"]
                log.evaluation.parameters = e.get("parameters", {})
                log.evaluation.metrics = e.get("metrics", {})
                log.evaluation.artifacts = e.get("artifacts", {})
                if "timing" in e:
                    log.evaluation.timing.start_time = e["timing"].get("start_time")
                    log.evaluation.timing.end_time = e["timing"].get("end_time")
                    log.evaluation.timing.duration_seconds = e["timing"].get("duration_seconds")
                    # Ustaw _start_ts na sztuczną wartość jeśli timing jest zakończony
                    if log.evaluation.timing.duration_seconds is not None:
                        log.evaluation.timing._start_ts = time.time() - log.evaluation.timing.duration_seconds
            
            log.add_note("Experiment log loaded and continued")
            return log
            
        except Exception as e:
            # Jeśli nie można wczytać, stwórz nowy
            log = ExperimentLog()
            log.add_note(f"Failed to load existing log: {e}")
            return log
    
    # ==================== Dataset Collection ====================
    
    def start_dataset_collection(self, seed: int, duration_sec: float, max_samples: int,
                                  scan_topic: str, odom_topic: str, gt_topic: str):
        """Rozpoczyna logowanie zbierania datasetu."""
        # Nie nadpisuj jeśli już zakończone
        if self.log.dataset.timing.is_completed():
            return
        self.log.dataset.timing.start()
        self.log.dataset.set_parameters(
            seed=seed, duration_sec=duration_sec, max_samples=max_samples,
            scan_topic=scan_topic, odom_topic=odom_topic, gt_topic=gt_topic
        )
        self.log.add_note("Dataset collection started")
        self.save()
    
    def end_dataset_collection(self, n_samples: int, scan_dim: int,
                                actual_duration_sec: float, file_path: str):
        """Kończy logowanie zbierania datasetu."""
        # Nie nadpisuj jeśli już zakończone
        if self.log.dataset.timing.is_completed():
            return
        self.log.dataset.timing.end()
        
        samples_per_second = n_samples / actual_duration_sec if actual_duration_sec > 0 else 0
        self.log.dataset.set_statistics(
            n_samples=n_samples, scan_dim=scan_dim,
            actual_duration_sec=actual_duration_sec,
            samples_per_second=samples_per_second
        )
        
        self.log.dataset.file_path = file_path
        if os.path.exists(file_path):
            self.log.dataset.file_size_mb = round(os.path.getsize(file_path) / (1024 * 1024), 3)
        
        self.log.add_note(f"Dataset collection completed: {n_samples} samples")
        self.save()
    
    # ==================== Training ====================
    
    def start_training(self, seed: int, max_epochs: int, patience: int,
                       min_delta: float, lr: float, val_ratio: float, batch_size: int):
        """Rozpoczyna logowanie treningu."""
        # Nie nadpisuj jeśli już zakończone
        if self.log.training.timing.is_completed():
            return
        self.log.training.timing.start()
        self.log.training.set_parameters(
            seed=seed, max_epochs=max_epochs, patience=patience,
            min_delta=min_delta, lr=lr, val_ratio=val_ratio, batch_size=batch_size
        )
        self.log.add_note("Training started")
        self.save()
    
    def set_training_dataset_info(self, n_total: int, n_train: int, n_val: int,
                                   input_dim: int, output_dim: int = 3):
        """Ustawia informacje o datasecie użytym do treningu."""
        self.log.training.set_dataset_info(
            n_total=n_total, n_train=n_train, n_val=n_val,
            input_dim=input_dim, output_dim=output_dim
        )
        self.save()
    
    def set_training_model_info(self, architecture: str, model):
        """Ustawia informacje o architekturze modelu."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log.training.set_model_info(
            architecture=architecture,
            total_params=total_params,
            trainable_params=trainable_params
        )
        self.save()
    
    def end_training(self, epochs_run: int, best_epoch: int, best_val_loss: float,
                     final_train_loss: float, early_stopped: bool,
                     model_path: str, history_path: str):
        """Kończy logowanie treningu."""
        self.log.training.timing.end()
        self.log.training.set_training_results(
            epochs_run=epochs_run, best_epoch=best_epoch,
            best_val_loss=best_val_loss, final_train_loss=final_train_loss,
            early_stopped=early_stopped
        )
        self.log.training.model_path = model_path
        self.log.training.history_path = history_path
        self.log.add_note(f"Training completed: {epochs_run} epochs, best_val_loss={best_val_loss:.6f}")
        self.save()
    
    # ==================== Inference ====================
    
    def start_inference(self, seed: int, scan_topic: str, odom_topic: str,
                        pose_topic: str, tf_parent: str, tf_child: str, model_path: str):
        """Rozpoczyna logowanie inferencji."""
        self.log.inference.timing.start()
        self.log.inference.set_parameters(
            seed=seed, scan_topic=scan_topic, odom_topic=odom_topic,
            pose_topic=pose_topic, tf_parent=tf_parent, tf_child=tf_child
        )
        self.log.inference.model_path = model_path
        self.log.add_note("Inference started")
        self.save()
    
    def end_inference(self, n_predictions: int, total_duration_sec: float,
                      avg_inference_time_ms: float):
        """Kończy logowanie inferencji."""
        self.log.inference.timing.end()
        self.log.inference.set_statistics(
            n_predictions=n_predictions,
            total_duration_sec=total_duration_sec,
            avg_inference_time_ms=avg_inference_time_ms
        )
        self.log.add_note(f"Inference completed: {n_predictions} predictions")
        self.save()

    def update_inference_statistics(self, n_predictions: int, total_duration_sec: float,
                                    avg_inference_time_ms: float):
        """Aktualizuje statystyki inferencji bez oznaczania etapu jako zakończonego."""
        self.log.inference.set_statistics(
            n_predictions=n_predictions,
            total_duration_sec=total_duration_sec,
            avg_inference_time_ms=avg_inference_time_ms
        )
        self.save()
    
    # ==================== Evaluation ====================
    
    def start_evaluation(self, seed: int, mode: str, duration_sec: float,
                         reference_map_yaml: str):
        """Rozpoczyna logowanie ewaluacji."""
        self.log.evaluation.timing.start()
        self.log.evaluation.set_parameters(
            seed=seed, mode=mode, duration_sec=duration_sec,
            reference_map_yaml=reference_map_yaml
        )
        self.log.add_note("Evaluation started")
        self.save()
    
    def end_evaluation(self, rmse_xy_baseline: float, rmse_theta_baseline: float,
                       rmse_xy_ai: Optional[float], rmse_theta_ai: Optional[float],
                       iou_map_baseline: Optional[float], iou_map_ai: Optional[float],
                       n_samples: int, artifacts: Dict[str, str],
                       iou_map_robak: Optional[float] = None,
                       iou_map_rywak: Optional[float] = None):
        """Kończy logowanie ewaluacji."""
        self.log.evaluation.timing.end()
        self.log.evaluation.set_metrics(
            rmse_xy_baseline=rmse_xy_baseline, rmse_theta_baseline=rmse_theta_baseline,
            rmse_xy_ai=rmse_xy_ai, rmse_theta_ai=rmse_theta_ai,
            iou_map_baseline=iou_map_baseline, iou_map_ai=iou_map_ai,
            n_samples=n_samples,
            iou_map_robak=iou_map_robak,
            iou_map_rywak=iou_map_rywak,
        )
        self.log.evaluation.artifacts = artifacts
        self.log.add_note("Evaluation completed")
        self.save()
    
    # ==================== Utilities ====================
    
    def add_note(self, note: str):
        """Dodaje notatkę do logu."""
        self.log.add_note(note)
        self.save()
    
    def finalize(self):
        """Finalizuje eksperyment i zapisuje całkowity czas."""
        if self.log.total_experiment_time_sec is not None:
            return
        self.log.total_experiment_time_sec = time.time() - self._experiment_start
        self.log.add_note(f"Experiment finalized. Total time: {self.log.total_experiment_time_sec:.1f}s")
        self.save()

    
    def _merge_timing(self, current: dict, saved: dict) -> dict:
        """Scala timing - zachowuje istniejące wartości, dodaje brakujące."""
        result = current.copy()
        for key in ["start_time", "end_time", "duration_seconds"]:
            if result.get(key) is None and saved.get(key) is not None:
                result[key] = saved[key]
        return result
    
    def _merge_dict(self, current: dict, saved: dict) -> dict:
        """Scala słowniki - zachowuje istniejące wartości, dodaje brakujące."""
        result = current.copy()
        for key, value in saved.items():
            if key not in result or result[key] is None or result[key] == {} or result[key] == []:
                result[key] = value
            elif isinstance(value, dict) and isinstance(result.get(key), dict):
                # Rekursywne scalanie dla zagnieżdżonych słowników
                if key == "timing":
                    result[key] = self._merge_timing(result[key], value)
                else:
                    result[key] = self._merge_dict(result[key], value)
        return result

    def _merge_top_level_metadata(self, current: dict, saved: dict) -> dict:
        """Zachowuje dodatkowe pola top-level spoza standardowego modelu logu."""
        result = current.copy()
        known_keys = {
            "experiment_id",
            "created_at",
            "system_info",
            "dataset",
            "training",
            "inference",
            "evaluation",
            "total_experiment_time_sec",
            "notes",
        }
        for key, value in saved.items():
            if key in known_keys:
                continue
            if key not in result or result[key] in (None, {}, []):
                result[key] = value
            elif isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = self._merge_dict(result[key], value)
        return result

    def save(self):
        """
        Zapisuje log do pliku JSON w sposób bezpieczny dla wielu procesów (wiele node'ów).

        Naprawia race-condition na `experiment_metadata.json.tmp`:
        - używa lockfile, aby serializować zapis między procesami,
        - używa unikalnego pliku tymczasowego (per PID), aby nie nadpisywać się nawzajem,
        - robi atomowy replace na końcu.
        """
        import fcntl
        import tempfile

        os.makedirs(self.out_dir, exist_ok=True)

        lock_path = self.metadata_path + ".lock"
        current_data = self.log.to_dict()

        # Serializacja zapisu między procesami
        with open(lock_path, "w", encoding="utf-8") as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
            try:
                # Wczytaj najnowsze dane z dysku i scałuj
                if os.path.exists(self.metadata_path):
                    try:
                        with open(self.metadata_path, "r", encoding="utf-8") as f:
                            saved_data = json.load(f)

                        for section in ["dataset", "training", "inference", "evaluation"]:
                            if section in saved_data and saved_data[section]:
                                current_data[section] = self._merge_dict(
                                    current_data.get(section, {}) or {},
                                    saved_data[section],
                                )

                        # Notatki – unikaj duplikatów
                        existing_notes = list(saved_data.get("notes", []) or [])
                        existing_set = set(existing_notes)
                        for note in current_data.get("notes", []) or []:
                            if note not in existing_set:
                                existing_notes.append(note)
                                existing_set.add(note)
                        current_data["notes"] = existing_notes

                        # Zachowaj created_at jeśli już istnieje
                        if saved_data.get("created_at") and not current_data.get("created_at"):
                            current_data["created_at"] = saved_data["created_at"]

                        # Zachowaj total time jeśli już istnieje
                        if saved_data.get("total_experiment_time_sec") is not None and current_data.get("total_experiment_time_sec") is None:
                            current_data["total_experiment_time_sec"] = saved_data["total_experiment_time_sec"]

                        current_data = self._merge_top_level_metadata(current_data, saved_data)

                    except Exception:
                        # Jeśli metadata jest w trakcie zapisu / uszkodzona, nie zabijaj treningu
                        pass

                # Unikalny tmp per proces (nie wspólny .tmp!)
                fd, tmp_path = tempfile.mkstemp(
                    prefix="experiment_metadata.",
                    suffix=f".{os.getpid()}.tmp",
                    dir=self.out_dir,
                )
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as f:
                        json.dump(current_data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())

                    os.replace(tmp_path, self.metadata_path)
                finally:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass

            finally:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)
    
    def get_summary(self) -> str:
        """Zwraca czytelne podsumowanie eksperymentu."""
        lines = [
            "=" * 60,
            f"EXPERIMENT SUMMARY: {self.log.experiment_id}",
            "=" * 60,
            "",
            f"Created: {self.log.created_at}",
            f"Output directory: {self.out_dir}",
            "",
        ]
        
        # Dataset
        if self.log.dataset.timing.duration_seconds:
            lines.append("📊 DATASET COLLECTION:")
            lines.append(f"   Duration: {self.log.dataset.timing.duration_seconds:.1f}s")
            if self.log.dataset.statistics:
                lines.append(f"   Samples: {self.log.dataset.statistics.get('n_samples', 'N/A')}")
                lines.append(f"   Rate: {self.log.dataset.statistics.get('samples_per_second', 0):.1f} samples/s")
            lines.append("")
        
        # Training
        if self.log.training.timing.duration_seconds:
            lines.append("🧠 MODEL TRAINING:")
            lines.append(f"   Duration: {self.log.training.timing.duration_seconds:.1f}s")
            if self.log.training.training_results:
                r = self.log.training.training_results
                lines.append(f"   Epochs: {r.get('epochs_run', 'N/A')}")
                lines.append(f"   Best val_loss: {r.get('best_validation_loss', 'N/A'):.6f}")
            lines.append("")
        
        # Inference
        if self.log.inference.timing.duration_seconds:
            lines.append("⚡ INFERENCE:")
            lines.append(f"   Duration: {self.log.inference.timing.duration_seconds:.1f}s")
            if self.log.inference.statistics:
                s = self.log.inference.statistics
                lines.append(f"   Predictions: {s.get('n_predictions', 'N/A')}")
                lines.append(f"   Avg time: {s.get('avg_inference_time_ms', 0):.2f}ms")
            lines.append("")
        
        # Evaluation
        if self.log.evaluation.metrics:
            lines.append("📈 EVALUATION METRICS:")
            m = self.log.evaluation.metrics
            lines.append(f"   RMSE XY (baseline): {m.get('rmse_xy_baseline', 'N/A'):.4f}m")
            if m.get('rmse_xy_ai') is not None:
                lines.append(f"   RMSE XY (AI): {m.get('rmse_xy_ai'):.4f}m")
                if m.get('rmse_xy_improvement_percent') is not None:
                    lines.append(f"   Improvement: {m.get('rmse_xy_improvement_percent'):+.1f}%")
            lines.append("")
        
        # Total time
        if self.log.total_experiment_time_sec:
            lines.append(f"⏱️ TOTAL EXPERIMENT TIME: {self.log.total_experiment_time_sec:.1f}s")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def append_to_summary(self):
        """
        Dodaje podsumowanie eksperymentu do pliku CSV w katalogu bazowym.
        Plik: base_out_dir/experiments_summary.csv
        
        Najpierw wczytuje najnowsze dane z pliku metadata, aby mieć kompletne dane.
        """
        import csv
        
        summary_path = os.path.join(self.base_out_dir, "experiments_summary.csv")
        
        # Wczytaj najnowsze dane z pliku (mogły być zaktualizowane przez inne node'y)
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = self.log.to_dict()
        else:
            data = self.log.to_dict()
        
        # Wyciągnij dane z wczytanego pliku
        dataset = data.get("dataset", {})
        training = data.get("training", {})
        inference = data.get("inference", {})
        evaluation = data.get("evaluation", {})
        
        # Przygotuj dane do zapisu
        row = {
            "experiment_id": data.get("experiment_id", self.log.experiment_id),
            "created_at": data.get("created_at", self.log.created_at),
            # Dataset
            "dataset_samples": dataset.get("statistics", {}).get("n_samples"),
            "dataset_duration_sec": dataset.get("timing", {}).get("duration_seconds"),
            "dataset_seed": dataset.get("parameters", {}).get("seed"),
            # Training
            "train_epochs": training.get("training_results", {}).get("epochs_run"),
            "train_best_epoch": training.get("training_results", {}).get("best_epoch"),
            "train_val_loss": training.get("training_results", {}).get("best_validation_loss"),
            "train_duration_sec": training.get("timing", {}).get("duration_seconds"),
            "train_lr": training.get("parameters", {}).get("learning_rate"),  # Fixed: was 'lr'
            "train_batch_size": training.get("parameters", {}).get("batch_size"),
            "train_patience": training.get("parameters", {}).get("patience"),
            "train_max_epochs": training.get("parameters", {}).get("max_epochs"),
            # Model
            "model_params": training.get("model_info", {}).get("total_parameters"),
            "model_architecture": training.get("model_info", {}).get("architecture"),
            # Inference
            "infer_predictions": inference.get("statistics", {}).get("n_predictions"),
            "infer_avg_time_ms": inference.get("statistics", {}).get("avg_inference_time_ms"),
            "infer_duration_sec": inference.get("timing", {}).get("duration_seconds"),
            # Evaluation
            "eval_rmse_xy_baseline": evaluation.get("metrics", {}).get("rmse_xy_baseline"),
            "eval_rmse_xy_ai": evaluation.get("metrics", {}).get("rmse_xy_ai"),
            "eval_rmse_theta_baseline": evaluation.get("metrics", {}).get("rmse_theta_baseline"),
            "eval_rmse_theta_ai": evaluation.get("metrics", {}).get("rmse_theta_ai"),
            "eval_iou_baseline": evaluation.get("metrics", {}).get("iou_map_baseline"),
            "eval_iou_ai": evaluation.get("metrics", {}).get("iou_map_ai"),
            "eval_iou_robak": evaluation.get("metrics", {}).get("iou_map_robak"),
            "eval_iou_rywak": evaluation.get("metrics", {}).get("iou_map_rywak"),
            "eval_xy_improvement_pct": evaluation.get("metrics", {}).get("rmse_xy_improvement_percent"),
            "eval_theta_improvement_pct": evaluation.get("metrics", {}).get("rmse_theta_improvement_percent"),
            "eval_n_samples": evaluation.get("metrics", {}).get("n_evaluation_samples"),
            # Total
            "total_time_sec": data.get("total_experiment_time_sec"),
            "output_dir": self.out_dir,
        }
        
        # Sprawdź czy plik istnieje
        file_exists = os.path.exists(summary_path)
        
        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        
        self.log.add_note(f"Experiment added to summary: {summary_path}")
        return summary_path
