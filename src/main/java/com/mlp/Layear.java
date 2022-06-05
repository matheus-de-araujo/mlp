package com.mlp;

import java.util.Random;

public class Layear {
    private float[] weights;
    private float[] bias;

    Layear(int size) {
        Random random = new Random();
        this.weights = new float[size];
        this.bias = new float[size];

        for(int i = 0; i < size; i++) {
            this.weights[i] = random.nextFloat();
        }

        for(int i = 0; i < size; i++) {
            this.bias[i] = random.nextFloat();                
        }
    }

    public float[] getBias() {
        return bias;
    }

    public void setBias(float[] bias) {
        this.bias = bias;
    }

    public float[] getWeights() {
        return weights;
    }

    public void setWeights(float[] weights) {
        this.weights = weights;
    }
}