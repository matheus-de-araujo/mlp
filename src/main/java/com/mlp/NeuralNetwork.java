package com.mlp;

import java.util.ArrayList;

public class NeuralNetwork {

    private float output[];

    private ArrayList<Inputs> inputs;
    private int iterations;
    private Layear[] layears;


    public NeuralNetwork(ArrayList<Inputs> inputs, int iterations) {

        this.inputs = inputs;

        this.iterations = iterations;
        
        this.layears = new Layear[4];

        initLayears(4);

    }

    private void initLayears(int size) {

        for(int i = 0; i < size; i++) {

            Layear l = new Layear(4);

            this.layears[i] = l;
        }
    }

    public boolean train() {

        int count = 0;

        do {

            int sucess = 1;

            for(Inputs input : inputs) {
                for(Layear layer : this.layears) {
                    this.output = run(input, layer.getWeights(), layer.getBias());
    
                    // if(output == input.d1) {
                    // } else {
                    //     sucess = 0;
                    //     adjust(input);
                    // }
                }
            }
            
            if(sucess == 1) {
                System.out.println("Epoca " + (count + 1));
            }
            count++;
        } while(count < this.iterations);

        // System.out.println("Peso 1: " + this.weights[0] + " Peso 2: " + this.weights[1]);
        return true;
    }


    public float activationFunction(float x) {

        return x >= 0 ? -1 : 1;

    }

    public float[] run(Inputs input, float[][] weights, float[] bias) {

        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights.length; j++) {
        
                this.output[i] = input.x1 * weights[i][j]; 
            }
            this.output[i] = this.output[i] - bias[i]; 
        }

        return this.output;
    }

    public void adjust(Inputs input) {

        // float learningRate = 0.1f;

        // float[] newWeights = {
        //     this.weights[0] + (learningRate * this.output * input.x1),
        //     this.weights[1] + (learningRate * this.output * input.x2),
        // };

        // this.weights = newWeights;
    }


}