import os
from pathlib import Path

class Config:
    # Paths
    # src/config.py -> src -> transformer (root)
    RootDir = Path(__file__).resolve().parent.parent
    DataDir = str(RootDir / "data")
    ResultsDir = str(RootDir / "results")
    
    # Data
    SYMBOL_LIMIT = 100
    INTERVAL = "15m"
    LookbackWindow = 96  # 24 hours
    ForwardLook = 12     # 3 hours
    
    # Sideways detection
    SlopeThreshold = 0.0002 
    R2Threshold = 0.25      
    
    # Features
    NumVolumeBins = 64
    ContextFeatures = ['atr', 'volatility', 'avg_volume', 'trend_slope', 'entry_bin_norm']
    
    # Model
    EmbedDim = 128
    NumHeads = 4
    NumLayers = 4
    Dropout = 0.1
    ContextDim = 5
    
    # Training
    BatchSize = 128
    Epochs = 10
    LearningRate = 1e-3
    TrainSplit = 0.7
    ValSplit = 0.15
    
    # Training Optimizations
    UseSampling = True      
    MaxSamples = 2000000    
    GradientAccumSteps = 1  
    CheckpointEveryN = 5000 
    LogEveryN = 100         
    ResumeFromCheckpoint = False
    
    # Trading Simulation
    RiskRewardRatio = 1.0 
    StopLossMultiplierATR = 1.0
    
    # Advanced Optimizations
    NumWorkers = 8          
    PersistentWorkers = True
    PinMemory = True        

class Utils:
    @staticmethod
    def ensure_dirs():
        os.makedirs(Config.DataDir, exist_ok=True)
        os.makedirs(Config.ResultsDir, exist_ok=True)
