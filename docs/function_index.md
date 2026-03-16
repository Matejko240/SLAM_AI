# Indeks funkcji

Zestawienie wygenerowane automatycznie na podstawie plików `.py` w repozytorium.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py

### seed_all
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:8`
- Typ: `function`
- Wejście: `seed: int`
- Wyjście: `brak`
- Opis: Ustawia ziarno dla all. Korzysta m.in. z: seed, manual_seed, manual_seed_all.

### ensure_dir
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:26`
- Typ: `function`
- Wejście: `path: str`
- Wyjście: `brak`
- Opis: Zapewnia dir. Korzysta m.in. z: makedirs.

### wrap
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:30`
- Typ: `function`
- Wejście: `a: float`
- Wyjście: `float`
- Opis: Normalizuje wrap.

### parse_filter_mode
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:34`
- Typ: `function`
- Wejście: `value: str, default: str = 'any'`
- Wyjście: `str`
- Opis: Parsuje filter mode. Korzysta m.in. z: lower, strip, str.

### pose_delta
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:51`
- Typ: `function`
- Wejście: `prev_xyth, curr_xyth`
- Wyjście: `PoseDelta`
- Opis: Obsługuje delta. Korzysta m.in. z: wrap, PoseDelta, float.

### scan_delta_rms
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:63`
- Typ: `function`
- Wejście: `scan_prev: np.ndarray, scan_curr: np.ndarray`
- Wyjście: `float`
- Opis: Obsługuje delta rms. Korzysta m.in. z: float, asarray, sqrt.

### passes_motion_filter
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:72`
- Typ: `function`
- Wejście: `prev_xyth, curr_xyth, dt_sec: float | None = None, min_translation: float = 0.0, min_rotation: float = 0.0, min_time_gap_sec: float = 0.0, min_scan_delta_rms: float = 0.0, scan_delta_rms_value: float | None = None, mode: str = 'any'`
- Wyjście: `tuple[bool, PoseDelta]`
- Opis: Weryfikuje motion filter. Korzysta m.in. z: pose_delta, parse_filter_mode, append.

### yaw_from_quat
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:106`
- Typ: `function`
- Wejście: `q`
- Wyjście: `brak`
- Opis: Obsługuje from quat. Korzysta m.in. z: atan2.

### quat_from_yaw
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:112`
- Typ: `function`
- Wejście: `yaw`
- Wyjście: `brak`
- Opis: Obsługuje from yaw. Korzysta m.in. z: sin, cos.

### xytheta_from_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:118`
- Typ: `function`
- Wejście: `odom_msg`
- Wyjście: `brak`
- Opis: Obsługuje from odometrię. Korzysta m.in. z: float, yaw_from_quat.

### xytheta_from_pose_stamped
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:125`
- Typ: `function`
- Wejście: `ps`
- Wyjście: `brak`
- Opis: Obsługuje from pozę stamped. Korzysta m.in. z: float, yaw_from_quat.

### Normalizer.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:133`
- Typ: `method`
- Wejście: `self, mean: np.ndarray, std: np.ndarray`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: astype, maximum.

### Normalizer.apply
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/common.py:137`
- Typ: `method`
- Wejście: `self, x: np.ndarray`
- Wyjście: `np.ndarray`
- Opis: Obsługuje apply.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py

### DatasetRecorder.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:18`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### DatasetRecorder.wait_for_topics
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:86`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Czeka na for topics. Korzysta m.in. z: info, now, time.

### DatasetRecorder.on_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:103`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię.

### DatasetRecorder.on_gt
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:107`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje ground truth.

### DatasetRecorder.on_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:111`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: asarray, astype, xytheta_from_odom.

### DatasetRecorder.check_done
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:147`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Sprawdza done. Korzysta m.in. z: save_and_exit, warn, now.

### DatasetRecorder.save_and_exit
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:158`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Zapisuje and exit. Korzysta m.in. z: info, astype, asarray.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder.py:225`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, DatasetRecorder, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py

### _stamp_to_sec
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:27`
- Typ: `function`
- Wejście: `stamp`
- Wyjście: `float`
- Opis: Stempluje to sec. Korzysta m.in. z: float.

### _resample_to_360
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:31`
- Typ: `function`
- Wejście: `ranges: np.ndarray`
- Wyjście: `np.ndarray`
- Opis: Resample dowolnego N do 360 przez interpolację po kącie.

### _sanitize_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:44`
- Typ: `function`
- Wejście: `msg: LaserScan`
- Wyjście: `np.ndarray`
- Opis: Czyści skan. Korzysta m.in. z: asarray, astype, clip.

### _augment_scan_pair
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:61`
- Typ: `function`
- Wejście: `scan_prev: np.ndarray, scan_curr: np.ndarray, rng: np.random.Generator, noise_std_scale: float, cut_fraction: float, cut_max_points: int, range_min: float = 0.08, range_max: float = 10.0`
- Wyjście: `brak`
- Opis: Augmentacja inspirowana ALSAI: szum Gaussa + losowe maskowanie fragmentu skanu.

### _delta_pose
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:98`
- Typ: `function`
- Wejście: `prev_xyth: Tuple[float, float, float], curr_xyth: Tuple[float, float, float], label_frame: str`
- Wyjście: `Tuple[float, float, float]`
- Opis: Delta między dwoma pozami GT.

### DatasetRecorderRobak.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:128`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### DatasetRecorderRobak.on_gt
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:242`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje ground truth. Korzysta m.in. z: _stamp_to_sec, append, xytheta_from_pose_stamped.

### DatasetRecorderRobak._nearest_gt
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:247`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje ground truth. Korzysta m.in. z: min, popleft, abs.

### DatasetRecorderRobak.wait_for_topics
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:260`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Czeka na for topics. Korzysta m.in. z: len, now, info.

### DatasetRecorderRobak.on_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:273`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: _sanitize_scan, _stamp_to_sec, _nearest_gt.

### DatasetRecorderRobak.check_done
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:348`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Sprawdza done. Korzysta m.in. z: save_and_exit, now, get_clock.

### DatasetRecorderRobak.save_and_exit
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:355`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Zapisuje and exit. Korzysta m.in. z: astype, ensure_dir, info.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_robak.py:424`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, DatasetRecorderRobak, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py

### _stamp_to_sec
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:26`
- Typ: `function`
- Wejście: `stamp`
- Wyjście: `float`
- Opis: Stempluje to sec. Korzysta m.in. z: float.

### _resample_to_360
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:30`
- Typ: `function`
- Wejście: `ranges: np.ndarray`
- Wyjście: `np.ndarray`
- Opis: Przeskalowuje to 360. Korzysta m.in. z: int, linspace, astype.

### _sanitize_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:41`
- Typ: `function`
- Wejście: `msg: LaserScan`
- Wyjście: `np.ndarray`
- Opis: Czyści skan. Korzysta m.in. z: asarray, astype, clip.

### _interp_angle
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:58`
- Typ: `function`
- Wejście: `th0: float, th1: float, alpha: float`
- Wyjście: `float`
- Opis: Interpoluje angle. Korzysta m.in. z: wrap, float.

### DatasetRecorderRywak.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:64`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### DatasetRecorderRywak.on_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:160`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: _stamp_to_sec, xytheta_from_odom, float.

### DatasetRecorderRywak._nearest_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:168`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: min, popleft, abs.

### DatasetRecorderRywak._interpolated_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:182`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: len, float, _interp_angle.

### DatasetRecorderRywak._odom_at
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:220`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje at. Korzysta m.in. z: _nearest_odom, _interpolated_odom.

### DatasetRecorderRywak.wait_for_topics
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:232`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Czeka na for topics. Korzysta m.in. z: len, now, info.

### DatasetRecorderRywak.on_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:242`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: _sanitize_scan, _stamp_to_sec, _odom_at.

### DatasetRecorderRywak.check_done
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:312`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Sprawdza done. Korzysta m.in. z: save_and_exit, now, get_clock.

### DatasetRecorderRywak.save_and_exit
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:319`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Zapisuje and exit. Korzysta m.in. z: astype, ensure_dir, info.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/dataset_recorder_rywak.py:389`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, DatasetRecorderRywak, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py

### get_system_info
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:21`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `Dict[str, Any]`
- Opis: Pobiera informacje o systemie.

### TimingInfo.start
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:62`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Rozpoczyna pomiar czasu tylko jeśli nie był już rozpoczęty.

### TimingInfo.end
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:68`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Kończy pomiar czasu tylko jeśli nie był już zakończony.

### TimingInfo.is_completed
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:75`
- Typ: `method`
- Wejście: `self`
- Wyjście: `bool`
- Opis: Sprawdza czy pomiar został zakończony.

### DatasetMetadata.set_parameters
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:89`
- Typ: `method`
- Wejście: `self, seed: int, duration_sec: float, max_samples: int, scan_topic: str, odom_topic: str, gt_topic: str`
- Wyjście: `brak`
- Opis: Ustawia parameters.

### DatasetMetadata.set_statistics
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:100`
- Typ: `method`
- Wejście: `self, n_samples: int, scan_dim: int, actual_duration_sec: float, samples_per_second: float`
- Wyjście: `brak`
- Opis: Ustawia statistics.

### TrainingMetadata.set_parameters
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:121`
- Typ: `method`
- Wejście: `self, seed: int, max_epochs: int, patience: int, min_delta: float, lr: float, val_ratio: float, batch_size: int`
- Wyjście: `brak`
- Opis: Ustawia parameters.

### TrainingMetadata.set_dataset_info
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:133`
- Typ: `method`
- Wejście: `self, n_total: int, n_train: int, n_val: int, input_dim: int, output_dim: int`
- Wyjście: `brak`
- Opis: Ustawia dataset info.

### TrainingMetadata.set_model_info
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:143`
- Typ: `method`
- Wejście: `self, architecture: str, total_params: int, trainable_params: int`
- Wyjście: `brak`
- Opis: Ustawia model info.

### TrainingMetadata.set_training_results
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:150`
- Typ: `method`
- Wejście: `self, epochs_run: int, best_epoch: int, best_val_loss: float, final_train_loss: float, early_stopped: bool`
- Wyjście: `brak`
- Opis: Ustawia training results.

### InferenceMetadata.set_parameters
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:170`
- Typ: `method`
- Wejście: `self, seed: int, scan_topic: str, odom_topic: str, pose_topic: str, tf_parent: str, tf_child: str`
- Wyjście: `brak`
- Opis: Ustawia parameters.

### InferenceMetadata.set_statistics
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:181`
- Typ: `method`
- Wejście: `self, n_predictions: int, total_duration_sec: float, avg_inference_time_ms: float`
- Wyjście: `brak`
- Opis: Ustawia statistics.

### EvaluationMetadata.set_parameters
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:199`
- Typ: `method`
- Wejście: `self, seed: int, mode: str, duration_sec: float, reference_map_yaml: str`
- Wyjście: `brak`
- Opis: Ustawia parameters.

### EvaluationMetadata.set_metrics
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:208`
- Typ: `method`
- Wejście: `self, rmse_xy_baseline: float, rmse_theta_baseline: float, rmse_xy_ai: Optional[float], rmse_theta_ai: Optional[float], iou_map_baseline: Optional[float], iou_map_ai: Optional[float], n_samples: int, iou_map_robak: Optional[float] = None, iou_map_rywak: Optional[float] = None`
- Wyjście: `brak`
- Opis: Ustawia metrics. Korzysta m.in. z: round.

### ExperimentLog.to_dict
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:252`
- Typ: `method`
- Wejście: `self`
- Wyjście: `Dict[str, Any]`
- Opis: Konwertuje log do słownika.

### ExperimentLog.add_note
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:266`
- Typ: `method`
- Wejście: `self, note: str`
- Wyjście: `brak`
- Opis: Dodaje notatkę do logu.

### generate_experiment_id
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:272`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `str`
- Opis: Generuje unikalny identyfikator eksperymentu z timestampem (jako string z prefiksem).

### get_experiment_dir
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:277`
- Typ: `function`
- Wejście: `base_out_dir: str, experiment_id: str`
- Wyjście: `str`
- Opis: Zwraca ścieżkę do katalogu eksperymentu.

### ExperimentLogger.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:317`
- Typ: `method`
- Wejście: `self, base_out_dir: str, experiment_id: Optional[str] = None, create_subdir: bool = True`
- Wyjście: `brak`
- Opis: Inicjalizuje logger eksperymentu.

### ExperimentLogger._find_or_create_experiment_id
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:355`
- Typ: `method`
- Wejście: `self`
- Wyjście: `str`
- Opis: Szuka aktywnego eksperymentu w katalogu bazowym lub tworzy nowy ID.

### ExperimentLogger.get_output_dir
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:387`
- Typ: `method`
- Wejście: `self`
- Wyjście: `str`
- Opis: Zwraca ścieżkę do katalogu wyjściowego eksperymentu.

### ExperimentLogger._load_existing
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:391`
- Typ: `method`
- Wejście: `self`
- Wyjście: `ExperimentLog`
- Opis: Wczytuje istniejący log (do kontynuacji eksperymentu).

### ExperimentLogger.start_dataset_collection
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:472`
- Typ: `method`
- Wejście: `self, seed: int, duration_sec: float, max_samples: int, scan_topic: str, odom_topic: str, gt_topic: str`
- Wyjście: `brak`
- Opis: Rozpoczyna logowanie zbierania datasetu.

### ExperimentLogger.end_dataset_collection
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:486`
- Typ: `method`
- Wejście: `self, n_samples: int, scan_dim: int, actual_duration_sec: float, file_path: str`
- Wyjście: `brak`
- Opis: Kończy logowanie zbierania datasetu.

### ExperimentLogger.start_training
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:510`
- Typ: `method`
- Wejście: `self, seed: int, max_epochs: int, patience: int, min_delta: float, lr: float, val_ratio: float, batch_size: int`
- Wyjście: `brak`
- Opis: Rozpoczyna logowanie treningu.

### ExperimentLogger.set_training_dataset_info
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:524`
- Typ: `method`
- Wejście: `self, n_total: int, n_train: int, n_val: int, input_dim: int, output_dim: int = 3`
- Wyjście: `brak`
- Opis: Ustawia informacje o datasecie użytym do treningu.

### ExperimentLogger.set_training_model_info
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:533`
- Typ: `method`
- Wejście: `self, architecture: str, model`
- Wyjście: `brak`
- Opis: Ustawia informacje o architekturze modelu.

### ExperimentLogger.end_training
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:544`
- Typ: `method`
- Wejście: `self, epochs_run: int, best_epoch: int, best_val_loss: float, final_train_loss: float, early_stopped: bool, model_path: str, history_path: str`
- Wyjście: `brak`
- Opis: Kończy logowanie treningu.

### ExperimentLogger.start_inference
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:561`
- Typ: `method`
- Wejście: `self, seed: int, scan_topic: str, odom_topic: str, pose_topic: str, tf_parent: str, tf_child: str, model_path: str`
- Wyjście: `brak`
- Opis: Rozpoczyna logowanie inferencji.

### ExperimentLogger.end_inference
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:573`
- Typ: `method`
- Wejście: `self, n_predictions: int, total_duration_sec: float, avg_inference_time_ms: float`
- Wyjście: `brak`
- Opis: Kończy logowanie inferencji.

### ExperimentLogger.update_inference_statistics
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:585`
- Typ: `method`
- Wejście: `self, n_predictions: int, total_duration_sec: float, avg_inference_time_ms: float`
- Wyjście: `brak`
- Opis: Aktualizuje statystyki inferencji bez oznaczania etapu jako zakończonego.

### ExperimentLogger.start_evaluation
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:597`
- Typ: `method`
- Wejście: `self, seed: int, mode: str, duration_sec: float, reference_map_yaml: str`
- Wyjście: `brak`
- Opis: Rozpoczyna logowanie ewaluacji.

### ExperimentLogger.end_evaluation
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:608`
- Typ: `method`
- Wejście: `self, rmse_xy_baseline: float, rmse_theta_baseline: float, rmse_xy_ai: Optional[float], rmse_theta_ai: Optional[float], iou_map_baseline: Optional[float], iou_map_ai: Optional[float], n_samples: int, artifacts: Dict[str, str], iou_map_robak: Optional[float] = None, iou_map_rywak: Optional[float] = None`
- Wyjście: `brak`
- Opis: Kończy logowanie ewaluacji.

### ExperimentLogger.add_note
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:630`
- Typ: `method`
- Wejście: `self, note: str`
- Wyjście: `brak`
- Opis: Dodaje notatkę do logu.

### ExperimentLogger.finalize
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:635`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Finalizuje eksperyment i zapisuje całkowity czas.

### ExperimentLogger._merge_timing
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:644`
- Typ: `method`
- Wejście: `self, current: dict, saved: dict`
- Wyjście: `dict`
- Opis: Scala timing - zachowuje istniejące wartości, dodaje brakujące.

### ExperimentLogger._merge_dict
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:652`
- Typ: `method`
- Wejście: `self, current: dict, saved: dict`
- Wyjście: `dict`
- Opis: Scala słowniki - zachowuje istniejące wartości, dodaje brakujące.

### ExperimentLogger.save
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:666`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Zapisuje log do pliku JSON w sposób bezpieczny dla wielu procesów (wiele node'ów).

### ExperimentLogger.get_summary
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:744`
- Typ: `method`
- Wejście: `self`
- Wyjście: `str`
- Opis: Zwraca czytelne podsumowanie eksperymentu.

### ExperimentLogger.append_to_summary
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/experiment_logger.py:803`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Dodaje podsumowanie eksperymentu do pliku CSV w katalogu bazowym.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py

### MLP.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:22`
- Typ: `method`
- Wejście: `self, in_dim: int, out_dim: int`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, Sequential, Linear.

### MLP.forward
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:34`
- Typ: `method`
- Wejście: `self, x`
- Wyjście: `brak`
- Opis: Obsługuje forward. Korzysta m.in. z: net.

### InferNode.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:39`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### InferNode.periodic_save_stats
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:103`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Okresowo zapisuje statystyki inferencji do metadata.json.

### InferNode.try_load_model
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:116`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje load model. Korzysta m.in. z: load, int, MLP.

### InferNode._publish_passthrough_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:160`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Publikuje oryginalną odometrię jako odom_ai (passthrough mode przed załadowaniem modelu).

### InferNode.on_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:181`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: _publish_passthrough_odom.

### InferNode.on_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:188`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: asarray, astype, clip.

### InferNode.log_inference_stats
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:255`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Loguje statystyki inferencji przy zamykaniu node'a.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_node.py:282`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, InferNode, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py

### _stamp_to_sec
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:22`
- Typ: `function`
- Wejście: `stamp`
- Wyjście: `float`
- Opis: Stempluje to sec. Korzysta m.in. z: float.

### _interp_angle
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:26`
- Typ: `function`
- Wejście: `th0: float, th1: float, alpha: float`
- Wyjście: `float`
- Opis: Interpoluje angle. Korzysta m.in. z: wrap, float.

### _delta_local
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:31`
- Typ: `function`
- Wejście: `prev_xyth, cur_xyth`
- Wyjście: `brak`
- Opis: Obsługuje local. Korzysta m.in. z: cos, sin, wrap.

### _resample_to_360
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:44`
- Typ: `function`
- Wejście: `ranges: np.ndarray`
- Wyjście: `np.ndarray`
- Opis: Przeskalowuje to 360. Korzysta m.in. z: int, linspace, astype.

### _sanitize_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:55`
- Typ: `function`
- Wejście: `msg: LaserScan`
- Wyjście: `np.ndarray`
- Opis: Czyści skan. Korzysta m.in. z: asarray, astype, clip.

### RobakConv1D.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:70`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, Sequential, Conv1d.

### RobakConv1D.forward
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:88`
- Typ: `method`
- Wejście: `self, x`
- Wyjście: `brak`
- Opis: Obsługuje forward. Korzysta m.in. z: head, feat.

### InferRobakNode.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:93`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### InferRobakNode.on_gt
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:197`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje ground truth. Korzysta m.in. z: xytheta_from_pose_stamped.

### InferRobakNode.on_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:203`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: _stamp_to_sec, xytheta_from_odom, append.

### InferRobakNode._nearest_odom_xyth
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:212`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje odometrię xyth. Korzysta m.in. z: min, popleft, abs.

### InferRobakNode.try_load_model
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:224`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje load model. Korzysta m.in. z: load, RobakConv1D, load_state_dict.

### InferRobakNode.periodic_save_stats
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:255`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje save stats. Korzysta m.in. z: time, float, end_inference.

### InferRobakNode.on_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:267`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: _sanitize_scan, _stamp_to_sec, max.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_robak_node.py:395`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, InferRobakNode, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py

### _stamp_to_sec
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:23`
- Typ: `function`
- Wejście: `stamp`
- Wyjście: `float`
- Opis: Stempluje to sec. Korzysta m.in. z: float.

### _resample_to_360
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:27`
- Typ: `function`
- Wejście: `ranges: np.ndarray`
- Wyjście: `np.ndarray`
- Opis: Przeskalowuje to 360. Korzysta m.in. z: int, linspace, astype.

### _sanitize_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:38`
- Typ: `function`
- Wejście: `msg: LaserScan`
- Wyjście: `np.ndarray`
- Opis: Czyści skan. Korzysta m.in. z: asarray, astype, clip.

### _interp_angle
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:52`
- Typ: `function`
- Wejście: `th0: float, th1: float, alpha: float`
- Wyjście: `float`
- Opis: Interpoluje angle. Korzysta m.in. z: wrap, float.

### _parse_hidden_dims
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:57`
- Typ: `function`
- Wejście: `value`
- Wyjście: `List[int]`
- Opis: Parsuje hidden dims. Korzysta m.in. z: int, list.

### MLP2.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:65`
- Typ: `method`
- Wejście: `self, in_dim: int, out_dim: int = 2, hidden_dims: List[int] = None, dropout: float = 0.0`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, int, append.

### MLP2.forward
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:82`
- Typ: `method`
- Wejście: `self, x`
- Wyjście: `brak`
- Opis: Obsługuje forward. Korzysta m.in. z: net.

### InferRywakNode.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:87`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### InferRywakNode.on_init_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:214`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje init odometrię. Korzysta m.in. z: xytheta_from_odom.

### InferRywakNode.on_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:220`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: _stamp_to_sec, xytheta_from_odom, float.

### InferRywakNode._nearest_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:230`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: min, popleft, abs.

### InferRywakNode._interpolated_odom
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:242`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: len, float, _interp_angle.

### InferRywakNode._odom_at
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:280`
- Typ: `method`
- Wejście: `self, t_scan: float`
- Wyjście: `brak`
- Opis: Obsługuje at. Korzysta m.in. z: _nearest_odom, _interpolated_odom.

### InferRywakNode.try_load_model
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:291`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje load model. Korzysta m.in. z: load, int, _parse_hidden_dims.

### InferRywakNode.periodic_save_stats
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:325`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje save stats. Korzysta m.in. z: time, float, end_inference.

### InferRywakNode.on_scan
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:337`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: _sanitize_scan, _stamp_to_sec, _odom_at.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/infer_rywak_node.py:487`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, InferRywakNode, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model.py

### MLP.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model.py:18`
- Typ: `method`
- Wejście: `self, in_dim: int, out_dim: int`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, Sequential, Linear.

### MLP.forward
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model.py:30`
- Typ: `method`
- Wejście: `self, x`
- Wyjście: `brak`
- Opis: Obsługuje forward. Korzysta m.in. z: net.

### TrainModel.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model.py:35`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### TrainModel.run_once
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model.py:84`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Uruchamia once. Korzysta m.in. z: time, info, start_training.

### TrainModel._save_zero_model
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model.py:261`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Zapisuje zero model. Korzysta m.in. z: MLP, zeros, ones.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model.py:282`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, TrainModel, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_robak.py

### RobakConv1D.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_robak.py:19`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, Sequential, Conv1d.

### RobakConv1D.forward
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_robak.py:37`
- Typ: `method`
- Wejście: `self, x`
- Wyjście: `brak`
- Opis: Obsługuje forward. Korzysta m.in. z: feat, head.

### TrainModelRobak.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_robak.py:44`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### TrainModelRobak.run_once
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_robak.py:91`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Uruchamia once. Korzysta m.in. z: time, info, load.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_robak.py:258`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, TrainModelRobak, spin.

## ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_rywak.py

### _parse_hidden_dims
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_rywak.py:18`
- Typ: `function`
- Wejście: `value`
- Wyjście: `List[int]`
- Opis: Parsuje hidden dims. Korzysta m.in. z: int, list.

### MLP2.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_rywak.py:26`
- Typ: `method`
- Wejście: `self, in_dim: int, out_dim: int = 2, hidden_dims: List[int] = None, dropout: float = 0.0`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, int, append.

### MLP2.forward
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_rywak.py:43`
- Typ: `method`
- Wejście: `self, x`
- Wyjście: `brak`
- Opis: Obsługuje forward. Korzysta m.in. z: net.

### TrainModelRywak.__init__
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_rywak.py:48`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### TrainModelRywak.run_once
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_rywak.py:111`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Uruchamia once. Korzysta m.in. z: time, info, load.

### main
- Plik: `ai_slam_ws/src/ai_slam_ai/ai_slam_ai/train_model_rywak.py:301`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, TrainModelRywak, spin.

## ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py

### parse_bool
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py:13`
- Typ: `function`
- Wejście: `value, default = False`
- Wyjście: `brak`
- Opis: Konwertuje parametr ROS (bool/str/int) do bool w przewidywalny sposób.

### AutoDriver.__init__
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py:30`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### AutoDriver.on_odom
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py:137`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: sqrt, len, append.

### AutoDriver._sector_stats
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py:179`
- Typ: `method`
- Wejście: `self, ranges: np.ndarray, angles: np.ndarray, a0: float, a1: float`
- Wyjście: `brak`
- Opis: Zwraca (min_robust, mean) w sektorze [a0, a1] (radiany), angle=0 przód, +pi/2 lewo, -pi/2 prawo.

### AutoDriver.on_scan
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py:201`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: now, asarray, int.

### AutoDriver.on_timer
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py:247`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje timer. Korzysta m.in. z: Twist, publish, warn.

### main
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/auto_driver.py:473`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, AutoDriver, spin.

## ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py

### _stamp_to_sec
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:10`
- Typ: `function`
- Wejście: `stamp`
- Wyjście: `float`
- Opis: Stempluje to sec. Korzysta m.in. z: float.

### GTPosePublisher.__init__
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:15`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, str.

### GTPosePublisher._frame_tokens
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:64`
- Typ: `method`
- Wejście: `frame_id: str`
- Wyjście: `brak`
- Opis: Obsługuje tokens. Korzysta m.in. z: lower, replace, strip.

### GTPosePublisher._tokens_match_hint
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:72`
- Typ: `method`
- Wejście: `tokens, hint: str`
- Wyjście: `bool`
- Opis: Obsługuje match hint. Korzysta m.in. z: lower, any, strip.

### GTPosePublisher._is_world_parent
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:78`
- Typ: `method`
- Wejście: `self, parent_tokens`
- Wyjście: `bool`
- Opis: Obsługuje world parent. Korzysta m.in. z: _tokens_match_hint.

### GTPosePublisher._match_score
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:87`
- Typ: `method`
- Wejście: `self, tr`
- Wyjście: `int`
- Opis: Obsługuje score. Korzysta m.in. z: _frame_tokens, _is_world_parent, _tokens_match_hint.

### GTPosePublisher._tf_brief
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:121`
- Typ: `method`
- Wejście: `msg: TFMessage, max_items: int = 8`
- Wyjście: `str`
- Opis: Obsługuje brief. Korzysta m.in. z: join, list, str.

### GTPosePublisher._select_unnamed_transform_by_odom
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:129`
- Typ: `method`
- Wejście: `self, msg: TFMessage`
- Wyjście: `brak`
- Opis: Fallback gdy bridge nie niesie nazw ramek (puste parent/child).

### GTPosePublisher._remember_last_world_pose
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:199`
- Typ: `method`
- Wejście: `self, stamp, pos`
- Wyjście: `brak`
- Opis: Obsługuje last world pozę. Korzysta m.in. z: _stamp_to_sec, float.

### GTPosePublisher._publish_pose
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:213`
- Typ: `method`
- Wejście: `self, stamp, frame_id: str, pos, quat`
- Wyjście: `brak`
- Opis: Obsługuje pozę. Korzysta m.in. z: PoseStamped, float, publish.

### GTPosePublisher.on_tf_world
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:226`
- Typ: `method`
- Wejście: `self, msg: TFMessage`
- Wyjście: `brak`
- Opis: Obsługuje TF world. Korzysta m.in. z: _publish_pose, _stamp_to_sec, _remember_last_world_pose.

### GTPosePublisher.on_odom
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:321`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: _publish_pose, float, _stamp_to_sec.

### main
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/gt_pose_publisher.py:350`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, GTPosePublisher, spin.

## ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py

### LifecycleManager.__init__
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py:11`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, list, create_timer.

### LifecycleManager._srv_name
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py:26`
- Typ: `method`
- Wejście: `self, node, suffix`
- Wyjście: `brak`
- Opis: Obsługuje name. Korzysta m.in. z: startswith.

### LifecycleManager._wait_service
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py:31`
- Typ: `method`
- Wejście: `self, client, timeout_sec`
- Wyjście: `brak`
- Opis: Czeka na service. Korzysta m.in. z: time, wait_for_service.

### LifecycleManager._get_state
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py:38`
- Typ: `method`
- Wejście: `self, node_name`
- Wyjście: `brak`
- Opis: Zwraca state. Korzysta m.in. z: create_client, Request, call_async.

### LifecycleManager._change_state
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py:51`
- Typ: `method`
- Wejście: `self, node_name, transition_id`
- Wyjście: `brak`
- Opis: Obsługuje state. Korzysta m.in. z: create_client, Request, int.

### LifecycleManager.tick
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py:65`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Wykonuje cykliczny krok dla tick. Korzysta m.in. z: _get_state, add, len.

### main
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/lifecycle_manager.py:106`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, LifecycleManager, spin.

## ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/odom_corruptor.py

### yaw_from_quat
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/odom_corruptor.py:13`
- Typ: `function`
- Wejście: `q`
- Wyjście: `brak`
- Opis: Obsługuje from quat. Korzysta m.in. z: atan2.

### quat_from_yaw
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/odom_corruptor.py:19`
- Typ: `function`
- Wejście: `yaw`
- Wyjście: `brak`
- Opis: Obsługuje from yaw. Korzysta m.in. z: sin, cos.

### wrap
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/odom_corruptor.py:25`
- Typ: `function`
- Wejście: `a`
- Wyjście: `brak`
- Opis: Normalizuje wrap.

### OdomCorruptor.__init__
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/odom_corruptor.py:30`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### OdomCorruptor.on_odom
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/odom_corruptor.py:57`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię. Korzysta m.in. z: max, yaw_from_quat, float.

### main
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/odom_corruptor.py:110`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, OdomCorruptor, spin.

## ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_fix.py

### ScanFix.__init__
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_fix.py:8`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, str.

### ScanFix.on_scan
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_fix.py:37`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: LaserScan, publish.

### main
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_fix.py:57`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, ScanFix, spin.

## ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py

### wrap
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:15`
- Typ: `function`
- Wejście: `a: float`
- Wyjście: `float`
- Opis: Normalizuje wrap.

### quat_from_yaw
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:19`
- Typ: `function`
- Wejście: `yaw: float`
- Wyjście: `brak`
- Opis: Obsługuje from yaw. Korzysta m.in. z: sin, cos.

### scan_to_points
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:25`
- Typ: `function`
- Wejście: `scan: LaserScan, max_use_range: float, max_points: int`
- Wyjście: `brak`
- Opis: LaserScan -> Nx2 points in laser frame.

### ScanMatcher.__init__
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:64`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, str.

### ScanMatcher._make_grid
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:179`
- Typ: `method`
- Wejście: `self, pts_prev: np.ndarray`
- Wyjście: `brak`
- Opis: Buduje siatkę bool dla poprzedniego skanu.

### ScanMatcher._score
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:192`
- Typ: `method`
- Wejście: `self, grid, size, pts_curr, dx, dy, dth`
- Wyjście: `brak`
- Opis: Score = hits - lambda * (dx^2 + dy^2 + (scale*dth)^2) Ruch (dx,dy,dth) jest rozumiany jako delta w układzie poprzedniej klatki.

### ScanMatcher._grid_search
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:220`
- Typ: `method`
- Wejście: `self, grid, size, pts_curr, center, win_xy, win_th, step_xy, step_th`
- Wyjście: `brak`
- Opis: Obsługuje search. Korzysta m.in. z: arange, _score, float.

### ScanMatcher._estimate_delta_local
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:241`
- Typ: `method`
- Wejście: `self, grid, size, pts_curr, center = None`
- Wyjście: `brak`
- Opis: 3-poziomowe przeszukiwanie wokół poprzedniej delty (albo 0).

### ScanMatcher._estimate_delta_bruteforce
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:271`
- Typ: `method`
- Wejście: `self, grid, size, pts_curr`
- Wyjście: `brak`
- Opis: Jedno duże przeszukanie (wolne).

### ScanMatcher._integrate_pose
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:282`
- Typ: `method`
- Wejście: `self, dx, dy, dth`
- Wyjście: `brak`
- Opis: SE2 compose: world_pose = world_pose ⊕ delta(prev_frame).

### ScanMatcher._local_to_world
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:289`
- Typ: `method`
- Wejście: `self, pts_local: np.ndarray, x: float, y: float, th: float`
- Wyjście: `np.ndarray`
- Opis: Obsługuje to world. Korzysta m.in. z: cos, sin, astype.

### ScanMatcher._world_to_local
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:296`
- Typ: `method`
- Wejście: `self, pts_world: np.ndarray, x: float, y: float, th: float`
- Wyjście: `np.ndarray`
- Opis: Obsługuje to local. Korzysta m.in. z: cos, sin, astype.

### ScanMatcher._estimate_delta_localmap
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:304`
- Typ: `method`
- Wejście: `self, pts_curr_local: np.ndarray`
- Wyjście: `brak`
- Opis: Obsługuje delta localmap. Korzysta m.in. z: concatenate, _world_to_local, _make_grid.

### ScanMatcher.on_scan
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:326`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan. Korzysta m.in. z: perf_counter, scan_to_points, float.

### main
- Plik: `ai_slam_ws/src/ai_slam_bringup/ai_slam_bringup/scan_matcher.py:433`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, ScanMatcher, spin.

## ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py

### generate_experiment_id
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:40`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `str`
- Opis: Generuje unikalny identyfikator eksperymentu.

### load_config
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:45`
- Typ: `function`
- Wejście: `config_file: str`
- Wyjście: `dict`
- Opis: Wczytuje konfigurację z pliku YAML.

### get_config_value
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:59`
- Typ: `function`
- Wejście: `config: dict, *keys, default = None`
- Wyjście: `brak`
- Opis: Bezpiecznie pobiera wartość z zagnieżdżonego słownika.

### parse_bool
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:70`
- Typ: `function`
- Wejście: `value, default = False`
- Wyjście: `brak`
- Opis: Konwertuje bool/str/int na bool w sposób odporny na 'false' jako string.

### merge_params
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:86`
- Typ: `function`
- Wejście: `*param_dicts`
- Wyjście: `brak`
- Opis: Łączy słowniki parametrów, ignorując wartości niebędące dict.

### extract_world_name
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:95`
- Typ: `function`
- Wejście: `world_path: str`
- Wyjście: `str`
- Opis: Próbuje odczytać nazwę świata z pliku SDF, fallback: nazwa pliku bez rozszerzenia.

### launch_setup
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:109`
- Typ: `function`
- Wejście: `context, *args, **kwargs`
- Wyjście: `brak`
- Opis: Funkcja setup wywoływana w runtime z dostępem do kontekstu.

### generate_launch_description
- Plik: `ai_slam_ws/src/ai_slam_bringup/launch/demo.launch.py:1194`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Generuje launch description. Korzysta m.in. z: LaunchDescription, DeclareLaunchArgument, OpaqueFunction.

## ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py

### wrap
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:28`
- Typ: `function`
- Wejście: `a`
- Wyjście: `brak`
- Opis: Normalizuje wrap.

### parse_filter_mode
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:32`
- Typ: `function`
- Wejście: `value: str, default: str = 'any'`
- Wyjście: `str`
- Opis: Parsuje filter mode. Korzysta m.in. z: lower, strip, str.

### pose_delta
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:49`
- Typ: `function`
- Wejście: `prev_xyth, curr_xyth`
- Wyjście: `PoseDelta`
- Opis: Obsługuje delta. Korzysta m.in. z: wrap, PoseDelta, float.

### passes_motion_filter
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:61`
- Typ: `function`
- Wejście: `prev_xyth, curr_xyth, dt_sec: float | None = None, min_translation: float = 0.0, min_rotation: float = 0.0, min_time_gap_sec: float = 0.0, mode: str = 'any'`
- Wyjście: `tuple[bool, PoseDelta]`
- Opis: Weryfikuje motion filter. Korzysta m.in. z: pose_delta, parse_filter_mode, append.

### yaw_from_quat
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:88`
- Typ: `function`
- Wejście: `q`
- Wyjście: `brak`
- Opis: Obsługuje from quat. Korzysta m.in. z: atan2.

### xytheta_from_pose
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:94`
- Typ: `function`
- Wejście: `ps: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje from pozę. Korzysta m.in. z: float, yaw_from_quat.

### xytheta_from_odom
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:101`
- Typ: `function`
- Wejście: `od: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje from odometrię. Korzysta m.in. z: float, yaw_from_quat.

### load_yaml_simple
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:108`
- Typ: `function`
- Wejście: `path: str`
- Wyjście: `dict`
- Opis: Wczytuje plik YAML simple. Korzysta m.in. z: open, strip, split.

### _read_token
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:121`
- Typ: `function`
- Wejście: `f`
- Wyjście: `brak`
- Opis: Czyta kolejny token z pliku PGM, pomija whitespace i komentarze #...

### load_pgm
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:143`
- Typ: `function`
- Wejście: `path: str`
- Wyjście: `np.ndarray`
- Opis: Wczytuje plik PGM. Korzysta m.in. z: open, _read_token, int.

### occgrid_to_array
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:183`
- Typ: `function`
- Wejście: `msg: OccupancyGrid`
- Wyjście: `np.ndarray`
- Opis: Obsługuje to array. Korzysta m.in. z: reshape, array.

### project_map_to_ref_grid
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:190`
- Typ: `function`
- Wejście: `ref_info: dict, ref_shape: tuple[int, int], slam_msg: OccupancyGrid`
- Wyjście: `brak`
- Opis: Rzutuje OccupancyGrid SLAM do siatki mapy referencyjnej (z orientacją obu map).

### map_iou
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:242`
- Typ: `function`
- Wejście: `ref_occ: np.ndarray, ref_info: dict, slam_msg: OccupancyGrid`
- Wyjście: `float`
- Opis: Mapuje IoU. Korzysta m.in. z: project_map_to_ref_grid, map_iou_binary.

### map_iou_binary
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:247`
- Typ: `function`
- Wejście: `ref_occ: np.ndarray, occ_s: np.ndarray, known: np.ndarray | None = None`
- Wyjście: `float`
- Opis: Mapuje IoU binary. Korzysta m.in. z: astype, int, ones_like.

### EvalNode.__init__
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:266`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: __init__, declare_parameter, int.

### EvalNode._load_ref_info
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:478`
- Typ: `method`
- Wejście: `self, yaml_path`
- Wyjście: `brak`
- Opis: Wczytuje mapę referencyjną info. Korzysta m.in. z: load_yaml_simple, strip, float.

### EvalNode._load_ref_occ
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:487`
- Typ: `method`
- Wejście: `self, yaml_path, info`
- Wyjście: `brak`
- Opis: Wczytuje mapę referencyjną occupancy grid. Korzysta m.in. z: dirname, join, load_pgm.

### EvalNode.on_gt
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:494`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje ground truth.

### EvalNode.on_odom
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:497`
- Typ: `method`
- Wejście: `self, msg: Odometry`
- Wyjście: `brak`
- Opis: Obsługuje odometrię.

### EvalNode.on_ai
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:500`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje AI. Korzysta m.in. z: len, info, get_logger.

### EvalNode.on_sm
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:508`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje sm.

### EvalNode.on_bf
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:511`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje bruteforce.

### EvalNode.on_robak
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:513`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje Robaka.

### EvalNode.on_rywak
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:516`
- Typ: `method`
- Wejście: `self, msg: PoseStamped`
- Wyjście: `brak`
- Opis: Obsługuje Rywaka.

### EvalNode.on_map
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:518`
- Typ: `method`
- Wejście: `self, msg: OccupancyGrid`
- Wyjście: `brak`
- Opis: Obsługuje mapę. Korzysta m.in. z: info, hasattr, get_logger.

### EvalNode.on_map_robak
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:523`
- Typ: `method`
- Wejście: `self, msg: OccupancyGrid`
- Wyjście: `brak`
- Opis: Obsługuje mapę Robaka.

### EvalNode.on_map_rywak
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:526`
- Typ: `method`
- Wejście: `self, msg: OccupancyGrid`
- Wyjście: `brak`
- Opis: Obsługuje mapę Rywaka.

### EvalNode.on_map_ai
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:528`
- Typ: `method`
- Wejście: `self, msg: OccupancyGrid`
- Wyjście: `brak`
- Opis: Obsługuje mapę AI. Korzysta m.in. z: info, hasattr, get_logger.

### EvalNode._stamp_points_to_ref_grid
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:533`
- Typ: `method`
- Wejście: `self, out_grid: np.ndarray, pose_msg: PoseStamped, scan_msg: LaserScan, state_key: str`
- Wyjście: `brak`
- Opis: Stempluje points to mapę referencyjną grid. Korzysta m.in. z: xytheta_from_pose, _stamp_to_sec, get.

### EvalNode.on_scan_points
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:601`
- Typ: `method`
- Wejście: `self, msg: LaserScan`
- Wyjście: `brak`
- Opis: Obsługuje skan points. Korzysta m.in. z: _stamp_points_to_ref_grid.

### EvalNode._stamp_to_sec
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:612`
- Typ: `method`
- Wejście: `self, stamp`
- Wyjście: `float`
- Opis: Stempluje to sec. Korzysta m.in. z: float.

### EvalNode._is_time_synced
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:615`
- Typ: `method`
- Wejście: `self, msg_a, msg_b`
- Wyjście: `bool`
- Opis: Obsługuje time synced. Korzysta m.in. z: _stamp_to_sec, abs.

### EvalNode.tick
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:625`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Wykonuje cykliczny krok dla tick. Korzysta m.in. z: xytheta_from_pose, xytheta_from_odom, append.

### EvalNode._request_maps_via_service
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:716`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Request maps via service call - more reliable than topic subscription.

### EvalNode._on_map_service_response
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:738`
- Typ: `method`
- Wejście: `self, future`
- Wyjście: `brak`
- Opis: Handle response from /slam_toolbox/dynamic_map service.

### EvalNode._on_map_ai_service_response
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:750`
- Typ: `method`
- Wejście: `self, future`
- Wyjście: `brak`
- Opis: Handle response from /slam_toolbox_ai/dynamic_map service.

### EvalNode.finish
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:762`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Domyka finish. Korzysta m.in. z: asarray, float, info.

### EvalNode._save_trajectory_data
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:986`
- Typ: `method`
- Wejście: `self, path`
- Wyjście: `brak`
- Opis: Zapisuje trajektorię data. Korzysta m.in. z: savez_compressed, asarray, reshape.

### EvalNode._reference_bounds_polygon
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:1036`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje bounds polygon. Korzysta m.in. z: float, array, cos.

### EvalNode._reference_walls_world_points
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:1076`
- Typ: `method`
- Wejście: `self, max_points: int = 50000`
- Wyjście: `brak`
- Opis: Zwraca punkty zajętych komórek mapy referencyjnej w układzie world.

### EvalNode._plot_trajectories
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:1105`
- Typ: `method`
- Wejście: `self, path`
- Wyjście: `brak`
- Opis: Rysuje trajectories. Korzysta m.in. z: asarray, _reference_bounds_polygon, _reference_walls_world_points.

### EvalNode._plot_errors
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:1182`
- Typ: `method`
- Wejście: `self, path`
- Wyjście: `brak`
- Opis: Rysuje errors. Korzysta m.in. z: asarray, figure, plot.

### EvalNode._plot_maps
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:1240`
- Typ: `method`
- Wejście: `self, path`
- Wyjście: `brak`
- Opis: Rysuje maps. Korzysta m.in. z: astype, _append_occ, len.

### main
- Plik: `ai_slam_ws/src/ai_slam_eval/ai_slam_eval/eval_node.py:1316`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: init, EvalNode, spin.

## scripts/generate_function_index.py

### rel_path
- Plik: `scripts/generate_function_index.py:112`
- Typ: `function`
- Wejście: `path: Path`
- Wyjście: `str`
- Opis: Obsługuje path. Korzysta m.in. z: str, relative_to, resolve.

### should_skip
- Plik: `scripts/generate_function_index.py:116`
- Typ: `function`
- Wejście: `path: Path`
- Wyjście: `bool`
- Opis: Obsługuje skip. Korzysta m.in. z: any.

### format_annotation
- Plik: `scripts/generate_function_index.py:120`
- Typ: `function`
- Wejście: `node: ast.AST | None`
- Wyjście: `str`
- Opis: Obsługuje annotation. Korzysta m.in. z: unparse.

### format_arg
- Plik: `scripts/generate_function_index.py:129`
- Typ: `function`
- Wejście: `arg: ast.arg, default_index: int | None = None, defaults: list[ast.AST] | None = None`
- Wyjście: `str`
- Opis: Obsługuje arg. Korzysta m.in. z: format_annotation, unparse.

### first_sentence
- Plik: `scripts/generate_function_index.py:141`
- Typ: `function`
- Wejście: `text: str`
- Wyjście: `str`
- Opis: Obsługuje sentence. Korzysta m.in. z: join, endswith, strip.

### get_call_names
- Plik: `scripts/generate_function_index.py:151`
- Typ: `function`
- Wejście: `node: ast.AST`
- Wyjście: `list[str]`
- Opis: Zwraca call names. Korzysta m.in. z: walk, isinstance, append.

### describe_from_name
- Plik: `scripts/generate_function_index.py:169`
- Typ: `function`
- Wejście: `name: str, node: ast.AST`
- Wyjście: `str`
- Opis: Obsługuje from name. Korzysta m.in. z: lstrip, get, strip.

### FunctionCollector.__init__
- Plik: `scripts/generate_function_index.py:189`
- Typ: `method`
- Wejście: `self, file_path: Path`
- Wyjście: `brak`
- Opis: Obsługuje init.

### FunctionCollector.visit_ClassDef
- Plik: `scripts/generate_function_index.py:195`
- Typ: `method`
- Wejście: `self, node: ast.ClassDef`
- Wyjście: `brak`
- Opis: Obsługuje ClassDef. Korzysta m.in. z: append, generic_visit, pop.

### FunctionCollector.visit_FunctionDef
- Plik: `scripts/generate_function_index.py:200`
- Typ: `method`
- Wejście: `self, node: ast.FunctionDef`
- Wyjście: `brak`
- Opis: Obsługuje FunctionDef. Korzysta m.in. z: _record_function.

### FunctionCollector.visit_AsyncFunctionDef
- Plik: `scripts/generate_function_index.py:203`
- Typ: `method`
- Wejście: `self, node: ast.AsyncFunctionDef`
- Wyjście: `brak`
- Opis: Obsługuje AsyncFunctionDef. Korzysta m.in. z: _record_function.

### FunctionCollector._record_function
- Plik: `scripts/generate_function_index.py:206`
- Typ: `method`
- Wejście: `self, node: ast.FunctionDef | ast.AsyncFunctionDef, kind: str`
- Wyjście: `brak`
- Opis: Obsługuje function. Korzysta m.in. z: generic_visit, enumerate, zip.

### collect_functions
- Plik: `scripts/generate_function_index.py:255`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `list[FunctionEntry]`
- Opis: Obsługuje functions. Korzysta m.in. z: sorted, rglob, should_skip.

### write_markdown
- Plik: `scripts/generate_function_index.py:270`
- Typ: `function`
- Wejście: `entries: list[FunctionEntry], output_path: Path`
- Wyjście: `brak`
- Opis: Obsługuje markdown. Korzysta m.in. z: mkdir, write_text, append.

### write_json
- Plik: `scripts/generate_function_index.py:296`
- Typ: `function`
- Wejście: `entries: list[FunctionEntry], output_path: Path`
- Wyjście: `brak`
- Opis: Obsługuje json. Korzysta m.in. z: mkdir, write_text, str.

### main
- Plik: `scripts/generate_function_index.py:306`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: ArgumentParser, add_argument, parse_args.

## scripts/generate_reference_map.py

### parse_pose
- Plik: `scripts/generate_reference_map.py:20`
- Typ: `function`
- Wejście: `text: str`
- Wyjście: `brak`
- Opis: pose: x y z roll pitch yaw (mogą być krótsze).

### compose_pose
- Plik: `scripts/generate_reference_map.py:29`
- Typ: `function`
- Wejście: `a, b`
- Wyjście: `brak`
- Opis: Składanie tylko 2D (x,y,yaw).

### extract_boxes_from_sdf
- Plik: `scripts/generate_reference_map.py:43`
- Typ: `function`
- Wejście: `sdf_path: str`
- Wyjście: `brak`
- Opis: Zwraca listę boxów z kolizji: [{x,y,yaw,sx,sy}].

### world_to_pixel
- Plik: `scripts/generate_reference_map.py:86`
- Typ: `function`
- Wejście: `x, y, origin_x, origin_y, resolution`
- Wyjście: `brak`
- Opis: Obsługuje to pixel. Korzysta m.in. z: int.

### draw_aabb
- Plik: `scripts/generate_reference_map.py:92`
- Typ: `function`
- Wejście: `grid, cx, cy, w, h, origin_x, origin_y, resolution, value = 0`
- Wyjście: `brak`
- Opis: Rysuje axis-aligned box (AABB).

### main
- Plik: `scripts/generate_reference_map.py:109`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: ArgumentParser, add_argument, parse_args.

## scripts/generate_thesis_report.py

### to_float
- Plik: `scripts/generate_thesis_report.py:27`
- Typ: `function`
- Wejście: `v`
- Wyjście: `brak`
- Opis: Obsługuje float. Korzysta m.in. z: float, isfinite.

### collect_results_paths
- Plik: `scripts/generate_thesis_report.py:39`
- Typ: `function`
- Wejście: `root_out: Path, sweep_paths, experiment_paths, result_paths`
- Wyjście: `brak`
- Opis: Obsługuje results paths. Korzysta m.in. z: set, resolve, exists.

### load_records
- Plik: `scripts/generate_thesis_report.py:83`
- Typ: `function`
- Wejście: `results_paths`
- Wyjście: `brak`
- Opis: Wczytuje records. Korzysta m.in. z: sort, append, get.

### write_experiment_table
- Plik: `scripts/generate_thesis_report.py:112`
- Typ: `function`
- Wejście: `records, out_path: Path`
- Wyjście: `brak`
- Opis: Obsługuje experiment table. Korzysta m.in. z: open, DictWriter, writeheader.

### stats
- Plik: `scripts/generate_thesis_report.py:143`
- Typ: `function`
- Wejście: `values`
- Wyjście: `brak`
- Opis: Obsługuje stats. Korzysta m.in. z: asarray, int, float.

### method_statistics
- Plik: `scripts/generate_thesis_report.py:164`
- Typ: `function`
- Wejście: `records`
- Wyjście: `brak`
- Opis: Obsługuje statistics. Korzysta m.in. z: stats, append, get.

### fmt
- Plik: `scripts/generate_thesis_report.py:231`
- Typ: `function`
- Wejście: `v, digits = 4`
- Wyjście: `brak`
- Opis: Obsługuje fmt.

### write_method_tables
- Plik: `scripts/generate_thesis_report.py:237`
- Typ: `function`
- Wejście: `rows, out_dir: Path`
- Wyjście: `brak`
- Opis: Obsługuje method tables. Korzysta m.in. z: open, DictWriter, writeheader.

### plot_box
- Plik: `scripts/generate_thesis_report.py:298`
- Typ: `function`
- Wejście: `records, metric_key, title, ylabel, out_path: Path`
- Wyjście: `brak`
- Opis: Rysuje box. Korzysta m.in. z: use, subplots, boxplot.

### plot_iou_bar
- Plik: `scripts/generate_thesis_report.py:327`
- Typ: `function`
- Wejście: `method_rows, out_path: Path`
- Wyjście: `brak`
- Opis: Rysuje IoU bar. Korzysta m.in. z: use, subplots, arange.

### plot_rmse_heatmap
- Plik: `scripts/generate_thesis_report.py:354`
- Typ: `function`
- Wejście: `records, out_path: Path`
- Wyjście: `brak`
- Opis: Rysuje RMSE heatmap. Korzysta m.in. z: full, enumerate, use.

### plot_mean_rank
- Plik: `scripts/generate_thesis_report.py:386`
- Typ: `function`
- Wejście: `records, out_path: Path`
- Wyjście: `brak`
- Opis: Rysuje mean rank. Korzysta m.in. z: zip, use, subplots.

### main
- Plik: `scripts/generate_thesis_report.py:429`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: ArgumentParser, add_argument, parse_args.

## scripts/generate_urdf_from_sdf.py

### text_or
- Plik: `scripts/generate_urdf_from_sdf.py:8`
- Typ: `function`
- Wejście: `elem, default = '0'`
- Wyjście: `brak`
- Opis: Obsługuje or. Korzysta m.in. z: strip.

### parse_pose
- Plik: `scripts/generate_urdf_from_sdf.py:14`
- Typ: `function`
- Wejście: `elem`
- Wyjście: `brak`
- Opis: Parsuje pozę. Korzysta m.in. z: split, len, text_or.

### geometry_to_urdf
- Plik: `scripts/generate_urdf_from_sdf.py:21`
- Typ: `function`
- Wejście: `geom_elem`
- Wyjście: `brak`
- Opis: Obsługuje to urdf. Korzysta m.in. z: find, text_or, Element.

### add_origin
- Plik: `scripts/generate_urdf_from_sdf.py:45`
- Typ: `function`
- Wejście: `parent, pose`
- Wyjście: `brak`
- Opis: Dodaje origin. Korzysta m.in. z: SubElement, set, join.

### add_inertial
- Plik: `scripts/generate_urdf_from_sdf.py:51`
- Typ: `function`
- Wejście: `link, inertial_elem`
- Wyjście: `brak`
- Opis: Dodaje inertial. Korzysta m.in. z: SubElement, parse_pose, add_origin.

### add_visual_or_collision
- Plik: `scripts/generate_urdf_from_sdf.py:74`
- Typ: `function`
- Wejście: `link, tag, elem`
- Wyjście: `brak`
- Opis: Dodaje visual or collision. Korzysta m.in. z: SubElement, parse_pose, add_origin.

### add_link
- Plik: `scripts/generate_urdf_from_sdf.py:90`
- Typ: `function`
- Wejście: `parent, sdf_link`
- Wyjście: `brak`
- Opis: Dodaje link. Korzysta m.in. z: get, SubElement, set.

### add_joint
- Plik: `scripts/generate_urdf_from_sdf.py:102`
- Typ: `function`
- Wejście: `parent, sdf_joint, child_pose`
- Wyjście: `brak`
- Opis: Dodaje joint. Korzysta m.in. z: get, SubElement, set.

### main
- Plik: `scripts/generate_urdf_from_sdf.py:131`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: dirname, join, parse.

## scripts/inspect_dataset.py

### load_reference_map
- Plik: `scripts/inspect_dataset.py:16`
- Typ: `function`
- Wejście: `yaml_path, pgm_path`
- Wyjście: `brak`
- Opis: Wczytuje mapę referencyjną z plików YAML i PGM.

### scan_to_points
- Plik: `scripts/inspect_dataset.py:55`
- Typ: `function`
- Wejście: `scan, pose, max_range = 5.0`
- Wyjście: `brak`
- Opis: Konwertuje skan LiDAR na punkty (x, y) w układzie globalnym.

### main
- Plik: `scripts/inspect_dataset.py:75`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: ArgumentParser, add_argument, parse_args.

## scripts/slam_dashboard.py

### read_json
- Plik: `scripts/slam_dashboard.py:60`
- Typ: `function`
- Wejście: `path: Path`
- Wyjście: `dict[str, Any]`
- Opis: Obsługuje json. Korzysta m.in. z: loads, read_text.

### read_text
- Plik: `scripts/slam_dashboard.py:67`
- Typ: `function`
- Wejście: `path: Path`
- Wyjście: `str`
- Opis: Obsługuje text. Korzysta m.in. z: read_text.

### safe_float
- Plik: `scripts/slam_dashboard.py:71`
- Typ: `function`
- Wejście: `value: str | None`
- Wyjście: `float | None`
- Opis: Obsługuje float. Korzysta m.in. z: float.

### safe_relative
- Plik: `scripts/slam_dashboard.py:80`
- Typ: `function`
- Wejście: `path: Path`
- Wyjście: `str`
- Opis: Obsługuje relative. Korzysta m.in. z: str, relative_to, resolve.

### list_config_files
- Plik: `scripts/slam_dashboard.py:87`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `list[dict[str, str]]`
- Opis: Obsługuje config files. Korzysta m.in. z: sorted, exists, glob.

### resolve_config_name
- Plik: `scripts/slam_dashboard.py:96`
- Typ: `function`
- Wejście: `name: str`
- Wyjście: `Path`
- Opis: Obsługuje config name. Korzysta m.in. z: resolve, startswith, ValueError.

### load_config_payload
- Plik: `scripts/slam_dashboard.py:105`
- Typ: `function`
- Wejście: `name: str`
- Wyjście: `dict[str, Any]`
- Opis: Wczytuje config payload. Korzysta m.in. z: resolve_config_name, read_text, exists.

### render_yaml_content
- Plik: `scripts/slam_dashboard.py:119`
- Typ: `function`
- Wejście: `parsed: Any`
- Wyjście: `str`
- Opis: Obsługuje plik YAML content. Korzysta m.in. z: safe_dump.

### save_config_payload
- Plik: `scripts/slam_dashboard.py:128`
- Typ: `function`
- Wejście: `name: str, content: str | None = None, parsed: Any | None = None`
- Wyjście: `dict[str, Any]`
- Opis: Zapisuje config payload. Korzysta m.in. z: resolve_config_name, write_text, render_yaml_content.

### ensure_function_index
- Plik: `scripts/slam_dashboard.py:144`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `None`
- Opis: Zapewnia function index. Korzysta m.in. z: run, exists, str.

### safe_resolve_local_path
- Plik: `scripts/slam_dashboard.py:161`
- Typ: `function`
- Wejście: `raw_path: str`
- Wyjście: `Path`
- Opis: Obsługuje resolve local path. Korzysta m.in. z: Path, unquote, is_absolute.

### load_trajectory_npz
- Plik: `scripts/slam_dashboard.py:172`
- Typ: `function`
- Wejście: `experiment_id: str`
- Wyjście: `np.lib.npyio.NpzFile`
- Opis: Wczytuje trajektorię plik NPZ. Korzysta m.in. z: load, exists, FileNotFoundError.

### wrap_angle_array
- Plik: `scripts/slam_dashboard.py:180`
- Typ: `function`
- Wejście: `values: np.ndarray`
- Wyjście: `np.ndarray`
- Opis: Normalizuje angle array. Korzysta m.in. z: asarray, astype.

### nearest_time_indices
- Plik: `scripts/slam_dashboard.py:185`
- Typ: `function`
- Wejście: `reference_time: np.ndarray, query_time: np.ndarray`
- Wyjście: `np.ndarray`
- Opis: Obsługuje time indices. Korzysta m.in. z: reshape, searchsorted, clip.

### get_pose_series
- Plik: `scripts/slam_dashboard.py:198`
- Typ: `function`
- Wejście: `data: np.lib.npyio.NpzFile, series_name: str`
- Wyjście: `tuple[np.ndarray, np.ndarray] | None`
- Opis: Zwraca pozę series. Korzysta m.in. z: get, reshape, asarray.

### get_error_series
- Plik: `scripts/slam_dashboard.py:216`
- Typ: `function`
- Wejście: `data: np.lib.npyio.NpzFile, series_name: str`
- Wyjście: `tuple[np.ndarray, np.ndarray, np.ndarray, str] | None`
- Opis: Zwraca error series. Korzysta m.in. z: get, get_pose_series, reshape.

### inspect_trajectory_capabilities
- Plik: `scripts/slam_dashboard.py:261`
- Typ: `function`
- Wejście: `exp_dir: Path`
- Wyjście: `dict[str, Any]`
- Opis: Obsługuje trajektorię capabilities. Korzysta m.in. z: exists, load, get_error_series.

### discover_experiments
- Plik: `scripts/slam_dashboard.py:283`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `list[dict[str, Any]]`
- Opis: Obsługuje experiments. Korzysta m.in. z: sorted, exists, glob.

### JobManager.__init__
- Plik: `scripts/slam_dashboard.py:353`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje init. Korzysta m.in. z: Lock, mkdir.

### JobManager.list_jobs
- Plik: `scripts/slam_dashboard.py:358`
- Typ: `method`
- Wejście: `self`
- Wyjście: `list[dict[str, Any]]`
- Opis: Obsługuje jobs. Korzysta m.in. z: sort, list, asdict.

### JobManager.read_log
- Plik: `scripts/slam_dashboard.py:364`
- Typ: `method`
- Wejście: `self, job_id: str, tail: int = 40000`
- Wyjście: `str`
- Opis: Obsługuje log. Korzysta m.in. z: Path, read_text, get.

### JobManager.start
- Plik: `scripts/slam_dashboard.py:375`
- Typ: `method`
- Wejście: `self, label: str, command: str, cwd: Path | None = None`
- Wyjście: `dict[str, Any]`
- Opis: Rozpoczyna start. Korzysta m.in. z: Job, Thread, start.

### JobManager._run
- Plik: `scripts/slam_dashboard.py:398`
- Typ: `method`
- Wejście: `self, job_id: str`
- Wyjście: `brak`
- Opis: Uruchamia run. Korzysta m.in. z: Path, time, open.

### command_for_training
- Plik: `scripts/slam_dashboard.py:432`
- Typ: `function`
- Wejście: `model_type: str, experiment_id: str`
- Wyjście: `tuple[str, str]`
- Opis: Obsługuje for training. Korzysta m.in. z: quote, ValueError, str.

### build_job_command
- Plik: `scripts/slam_dashboard.py:487`
- Typ: `function`
- Wejście: `payload: dict[str, Any]`
- Wyjście: `tuple[str, str]`
- Opis: Buduje job command. Korzysta m.in. z: strip, ValueError, safe_resolve_local_path.

### make_placeholder_figure
- Plik: `scripts/slam_dashboard.py:528`
- Typ: `function`
- Wejście: `title: str, message: str`
- Wyjście: `bytes`
- Opis: Obsługuje placeholder figure. Korzysta m.in. z: subplots, text, set_title.

### figure_to_png
- Plik: `scripts/slam_dashboard.py:536`
- Typ: `function`
- Wejście: `fig`
- Wyjście: `bytes`
- Opis: Obsługuje to png. Korzysta m.in. z: BytesIO, tight_layout, savefig.

### plot_trajectory_image
- Plik: `scripts/slam_dashboard.py:545`
- Typ: `function`
- Wejście: `experiment_id: str, series_names: list[str], x_min: float | None, x_max: float | None, y_min: float | None, y_max: float | None`
- Wyjście: `bytes`
- Opis: Rysuje trajektorię image. Korzysta m.in. z: subplots, set_title, set_xlabel.

### plot_error_image
- Plik: `scripts/slam_dashboard.py:591`
- Typ: `function`
- Wejście: `experiment_id: str, series_names: list[str], metric: str, time_min: float | None, time_max: float | None, y_min: float | None, y_max: float | None`
- Wyjście: `bytes`
- Opis: Rysuje error image. Korzysta m.in. z: startswith, subplots, set.

### DashboardHandler.do_GET
- Plik: `scripts/slam_dashboard.py:1900`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje GET. Korzysta m.in. z: urlparse, _send_text, ensure_function_index.

### DashboardHandler.do_POST
- Plik: `scripts/slam_dashboard.py:1997`
- Typ: `method`
- Wejście: `self`
- Wyjście: `brak`
- Opis: Obsługuje POST. Korzysta m.in. z: urlparse, _send_json, _send_text.

### DashboardHandler.log_message
- Plik: `scripts/slam_dashboard.py:2042`
- Typ: `method`
- Wejście: `self, format: str, *args`
- Wyjście: `brak`
- Opis: Obsługuje message.

### DashboardHandler._send_json
- Plik: `scripts/slam_dashboard.py:2045`
- Typ: `method`
- Wejście: `self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK`
- Wyjście: `brak`
- Opis: Obsługuje json. Korzysta m.in. z: encode, _send_bytes, dumps.

### DashboardHandler._send_text
- Plik: `scripts/slam_dashboard.py:2049`
- Typ: `method`
- Wejście: `self, text: str, status: HTTPStatus = HTTPStatus.OK, content_type: str = 'text/plain; charset=utf-8'`
- Wyjście: `brak`
- Opis: Obsługuje text. Korzysta m.in. z: _send_bytes, encode.

### DashboardHandler._send_bytes
- Plik: `scripts/slam_dashboard.py:2052`
- Typ: `method`
- Wejście: `self, data: bytes, status: HTTPStatus = HTTPStatus.OK, content_type: str = 'application/octet-stream'`
- Wyjście: `brak`
- Opis: Obsługuje bytes. Korzysta m.in. z: send_response, send_header, end_headers.

### main
- Plik: `scripts/slam_dashboard.py:2060`
- Typ: `function`
- Wejście: `brak`
- Wyjście: `brak`
- Opis: Stanowi punkt wejścia dla main. Korzysta m.in. z: ArgumentParser, add_argument, parse_args.
