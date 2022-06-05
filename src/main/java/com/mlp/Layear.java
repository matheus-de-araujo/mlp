package com.mlp;

import java.util.Random;

public class Layear {
    private float[][] weights;
    private float[] bias;
    private float[] output;
    private Inputs input;

    Layear(int size) {
        Random random = new Random();
        this.weights = new float[size][size];
        this.output = new float[size];
        this.bias = new float[size];

        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                this.weights[i][j] = random.nextFloat();
            }
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

    public float[][] getWeights() {
        return weights;
    }

    public void setWeights(float[][] weights) {
        this.weights = weights;
    }

    public float[] getOutput() {
        return output;
    }

    public void setOutput(float[] output) {
        this.output = output;
    }

    public Inputs getInput() {
        return input;
    }

    public void setInput(Inputs input) {
        this.input = input;
    }
}
