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
        
        this.output = new float[4];

        initLayears(this.output.length);

    }

    private void initLayears(int size) {

        for(int i = 0; i < size; i++) {

            Layear l = new Layear(4);

            this.layears[i] = l;
        }
    }


    private float[] activationFunction(float[] output) {

        for(int i = 0; i < output.length; i++) {
            output[i] = (float) (1 / (1 + Math.exp((double) (output[i] * -1))));
        }

        return output;

    }

    private float[] run(Inputs input, float[][] weights, float[] bias) {

        float[] arrInput = {input.x1, input.x2, input.x3, input.x4};

        for(int i = 0; i < weights.length; i++) {
            for(int j = 0; j < weights.length; j++) {
        
                this.output[i] += arrInput[j] * weights[i][j]; 
            }
            this.output[i] = this.output[i] - bias[i]; 
        }

        return activationFunction(this.output);
    }

    public boolean train() {

        int count = 0;

        do {

            for(Inputs input : inputs) {
                boolean firstLoop = true;
                for(Layear layer : this.layears) {
                    if(!firstLoop) {
                        setOutputInInput(input);
                    }
                    
                    this.output = run(input, layer.getWeights(), layer.getBias());
                    
                    firstLoop = false;
                }

                
                if(validateResult(input, output)) {
                } else {
                    setOutputInInput(input);
                    adjust(input);
                }
            }
            
            count++;
        } while(count < this.iterations);

        return true;
    }

    private boolean validateResult(Inputs input, float[] output) {
        return input.d1 == output[0] && input.d2 == output[1] && input.d3 == output[2];
    }

    private Inputs setOutputInInput(Inputs input) {
        input.x1 = output[0];
        input.x2 = output[1];
        input.x3 = output[2];
        input.x4 = output[3];

        return input;
    }

    private void adjust(Inputs input) {

        float learningRate = 0.1f;

        // float[] newWeights = {
        //     this.weights[0] + (learningRate * this.output * input.x1),
        //     this.weights[1] + (learningRate * this.output * input.x2),
        // };

        // this.weights = newWeights;
    }
}