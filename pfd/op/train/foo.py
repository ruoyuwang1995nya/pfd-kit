import os
import json
import time
from pathlib import Path
from typing import Tuple, Any, Dict, List
from dargs import Argument

from .train import Train


class TrainFoo(Train):
    
    default_optional_parameter = {
        "finetune_mode": "foo_fake",
        "fake_training_time": 5.0,  # seconds
        "fake_success": True,
        "create_fake_logs": True
    }
    
    model_name = "foo.pth"
    def _process_script(self, 
                        input_dict) -> Any:
        return super()._process_script(input_dict)
    
    @staticmethod
    def training_args():
        """Fake training specific arguments."""
        base_args = Train.training_args()
        
        foo_args = [
            Argument(
                "fake_training_time", 
                float, 
                optional=True, 
                default=5.0,
                doc="Time to simulate training (seconds)"
            ),
            Argument(
                "fake_success",
                bool,
                optional=True,
                default=True,
                doc="Whether to simulate successful training"
            ),
            Argument(
                "create_fake_logs",
                bool,
                optional=True,
                default=True,
                doc="Whether to create fake log files"
            ),
            Argument(
                "fake_model_name",
                str,
                optional=True,
                default="fake_model.pt",
                doc="Name of the fake output model file"
            )
        ]
        
        return base_args + foo_args
    
    def prepare_train(self, 
                     config: Dict[str, Any],
                     init_model: Path, 
                     init_data: Dict[str, Any], 
                     iter_data: Dict[str, Any],
                     valid_data: Dict[str, Any], 
                     optional_param: Dict[str, Any],
                     **kwargs) -> None:
        """
        Prepare fake training - just copy input model and create config files.
        
        Parameters
        ----------
        work_dir : Path
            Working directory for training
        task_path : Path 
            Task specific path
        config : Dict[str, Any]
            Training configuration
        init_model : Path
            Initial model file to pass through
        init_data : Dict[str, Any]
            Initial training data
        iter_data : Dict[str, Any]
            Iterative training data
        valid_data : Dict[str, Any]
            Validation data
        optional_param : Dict[str, Any]
            Optional parameters
        """
        print("=== RunTrainFoo: Preparing fake training ===")
        
        # Extract parameters
        fake_time = optional_param.get("fake_training_time", self.default_optional_parameter["fake_training_time"])
        fake_success = optional_param.get("fake_success", self.default_optional_parameter["fake_success"])
        create_logs = optional_param.get("create_fake_logs", self.default_optional_parameter["create_fake_logs"])
        self.model_name = optional_param.get("model_name", self.model_name)
        
        # Create fake config file
        fake_config = {
            "model_type": "FakeModel",
            "training_time": fake_time,
            "success_mode": fake_success,
            "input_model": str(init_model) if init_model else "none",
            "data_info": {
                "init_data_paths": len(init_data) if init_data else 0,
                "valid_data_paths": len(valid_data) if valid_data else 0
            }
        }
        
        # Write fake config
        with open("fake_train_config.json", "w") as f:
            json.dump(fake_config, f, indent=2)
        
        # Copy input model to output location (pass-through)
        if init_model and init_model.exists():
            import shutil
            output_model = Path(self.model_name)
            shutil.copy2(init_model, output_model)
            print(f"Copied input model {init_model} to {output_model}")
        else:
            # Create a fake model file
            with open(self.model_name, "w") as f:
                f.write("# Fake model file created by RunTrainFoo\n")
                f.write(f"# Created at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Config: {fake_config}\n")
            print(f"Created fake model file: {self.model_name}")
        
        # Store config for run_train method
        self._fake_config = fake_config
        
        print(f"Fake training preparation completed")
        print(f"- Training time: {fake_time}s")
        print(f"- Success mode: {fake_success}")
        print(f"- Create logs: {create_logs}")
    
    def run_train(self) -> Tuple[str, str, str]:
        """
        Execute fake training - simulate training time and create logs.
        
        Returns
        -------
        Tuple[str, int, str, str]
            (model_file, return_code, stdout, stderr)
        """
        print("=== RunTrainFoo: Starting fake training ===")
        
        # Get config from prepare_train
        config = getattr(self, '_fake_config', {})
        model_name = getattr(self, 'model_name', 'fake_model.pt')
        
        fake_time = config.get("training_time", 5.0)
        fake_success = config.get("success_mode", True)
        
        # Create fake log files
        self._create_fake_logs(config, fake_success)
        
        # Simulate training time
        print(f"Simulating training for {fake_time} seconds...")
        time.sleep(fake_time)
        
        # Generate fake training output
        if fake_success:
            stdout = self._generate_success_log()
            stderr = ""
            #return_code = 0
            print("Fake training completed successfully!")
        else:
            stdout = self._generate_failure_log()
            stderr = "Fake training failed as requested"
            #return_code = 1
            print("Fake training failed as requested!")
        
        return model_name, stdout, stderr

    def write_log(self, 
                  ret: str, 
                  out: str, 
                  err: str) -> None:
        """Write log content to specified log file."""
        with open(self.log_file, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {out}\n")
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {err}\n")

    def normalize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize configuration for fake training."""
        normalized = config.copy()
        
        # Set fake training specific defaults
        if "fake_training_time" not in normalized:
            normalized["fake_training_time"] = self.default_optional_parameter["fake_training_time"]
        if "fake_success" not in normalized:
            normalized["fake_success"] = self.default_optional_parameter["fake_success"] 
        if "create_fake_logs" not in normalized:
            normalized["create_fake_logs"] = self.default_optional_parameter["create_fake_logs"]
            
        return normalized
    
    def _create_fake_logs(self, config: Dict[str, Any], success: bool) -> None:
        """Create fake log files to simulate real training."""
        
        # Create fake training log
        with open("training.log", "w") as f:
            f.write("=== Fake Training Log ===\n")
            f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Configuration: {json.dumps(config, indent=2)}\n")
            f.write("\n=== Training Progress ===\n")
            
            # Simulate training epochs
            num_epochs = 10
            for epoch in range(num_epochs):
                train_loss = 1.0 - (epoch * 0.08) + (0.01 * (epoch % 3))  # Fake decreasing loss
                val_loss = train_loss + 0.05
                f.write(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}\n")
                
                if not success and epoch == 7:  # Simulate failure at epoch 8
                    f.write(f"ERROR: Fake training failure at epoch {epoch+1}\n")
                    break
            
            if success:
                f.write("\n=== Training Completed Successfully ===\n")
                f.write("Best model saved: fake_model.pt\n")
            else:
                f.write("\n=== Training Failed ===\n")
                f.write("Training terminated due to simulated error\n")
            
            f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Create fake metrics file
        with open("metrics.json", "w") as f:
            if success:
                metrics = {
                    "final_train_loss": 0.234,
                    "final_val_loss": 0.278,
                    "best_epoch": 8,
                    "total_epochs": 10,
                    "training_time": config.get("training_time", 5.0),
                    "converged": True
                }
            else:
                metrics = {
                    "final_train_loss": 0.456,
                    "final_val_loss": 0.523,
                    "failed_at_epoch": 8,
                    "total_epochs": 8,
                    "training_time": config.get("training_time", 5.0),
                    "converged": False,
                    "error": "Simulated training failure"
                }
            json.dump(metrics, f, indent=2)
        
        # Create fake checkpoint directory
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        for i in [1, 5, 10] if success else [1, 5, 8]:
            checkpoint_file = checkpoint_dir / f"checkpoint_epoch_{i}.pt"
            with open(checkpoint_file, "w") as f:
                f.write(f"# Fake checkpoint at epoch {i}\n")
                f.write(f"# Loss: {1.0 - i*0.08:.4f}\n")
    
    def _generate_success_log(self) -> str:
        """Generate fake successful training stdout."""
        return """
=== Fake Training Output ===
Initializing fake model...
Loading training data...
Starting training loop...

Epoch   1/10: loss=0.920, val_loss=0.970, time=0.5s
Epoch   2/10: loss=0.840, val_loss=0.890, time=0.5s
Epoch   3/10: loss=0.770, val_loss=0.820, time=0.5s
Epoch   4/10: loss=0.680, val_loss=0.730, time=0.5s
Epoch   5/10: loss=0.600, val_loss=0.650, time=0.5s
Epoch   6/10: loss=0.520, val_loss=0.570, time=0.5s
Epoch   7/10: loss=0.440, val_loss=0.490, time=0.5s
Epoch   8/10: loss=0.360, val_loss=0.410, time=0.5s ← best
Epoch   9/10: loss=0.350, val_loss=0.420, time=0.5s
Epoch  10/10: loss=0.340, val_loss=0.430, time=0.5s

Training completed successfully!
Best model saved to fake_model.pt
Final validation loss: 0.410
"""
    
    def _generate_failure_log(self) -> str:
        """Generate fake failed training stdout."""
        return """
=== Fake Training Output ===
Initializing fake model...
Loading training data...
Starting training loop...

Epoch   1/10: loss=0.920, val_loss=0.970, time=0.5s
Epoch   2/10: loss=0.840, val_loss=0.890, time=0.5s
Epoch   3/10: loss=0.770, val_loss=0.820, time=0.5s
Epoch   4/10: loss=0.680, val_loss=0.730, time=0.5s
Epoch   5/10: loss=0.600, val_loss=0.650, time=0.5s
Epoch   6/10: loss=0.520, val_loss=0.570, time=0.5s
Epoch   7/10: loss=0.440, val_loss=0.490, time=0.5s
Epoch   8/10: FAILED - Simulated training error

ERROR: Fake training failure occurred
Training terminated at epoch 8
"""

