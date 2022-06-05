package com.mlp;

import java.util.ArrayList;
// import java.util.Random;

public class NeuralNetwork {

    private float[] weights;
    private float bias;
    private float output;

    private ArrayList<Inputs> inputs;
    private int iterations;

    public NeuralNetwork(ArrayList<Inputs> inputs, int iterations) {

        this.inputs = inputs;

        this.iterations = iterations;

        this.weights = new float[inputs.size()];

        this.initWeights();

    }

    private void initWeights() {
        // Random random = new Random();

        for(int i = 0; i < this.weights.length; i++) {

            // this.weights[i] = random.nextFloat();
            this.weights[i] = 0;

            // System.out.println("w: " + this.weights[i]);

        }

        this.bias = 0;
        //System.out.println("b: " + this.bias);

    }

    public float activationFunction(float x) {

        return x >= 0 ? -1 : 1;

    }

    public float run(Inputs input) {

        float sum = input.x1 * this.weights[0] + input.x2 * this.weights[1];

        sum += this.bias;

        return this.activationFunction(sum);
    }

    public void adjust(Inputs input) {

        float learningRate = 0.5f;

        float[] newWeights = {
            this.weights[0] + (learningRate * this.output * input.x1),
            this.weights[1] + (learningRate * this.output * input.x2),
        };

        this.weights = newWeights;
    }

    public boolean train() {

        int count = 0;

        do {

            int sucess = 1;

            for(Inputs input : inputs) {
                this.output = run(input);

                if(output == input.d) {
                } else {
                    sucess = 0;
                    adjust(input);
                }
            }
            
            if(sucess == 1) {
                System.out.println("Epoca " + (count + 1));
            }
            count++;
        } while(count < this.iterations);

        System.out.println("Peso 1: " + this.weights[0] + " Peso 2: " + this.weights[1]);
        return true;
    }

}