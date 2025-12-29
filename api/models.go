package main

import (
	"time"
)

type Hyperparameters struct {
	GateOptimizer  string  `json:"gateOptimizer" bson:"gateOptimizer"`
	Gates          int     `json:"gates" bson:"gates"`
	NetworkLayers  int     `json:"networkLayers" bson:"networkLayers"`
	Grouping       int     `json:"grouping" bson:"grouping"`
	GroupSumTau    int     `json:"groupSumTau" bson:"groupSumTau"`
	ResidualLayers int     `json:"residualLayers" bson:"residualLayers"`
	NoiseTemp      float64 `json:"noiseTemp" bson:"noiseTemp"`
	Epochs         int     `json:"epochs" bson:"epochs"`
}

type job struct {
	TimeCreated     time.Time       `json:"timeCreated" bson:"timeCreated"`
	TotalParameters int             `json:"totalParameters" bson:"totalParameters"`
	Hyperparameters Hyperparameters `json:"hyperparameters" bson:"hyperparameters"`
	Status          string          `json:"status" bson:"status"`
}
