package com.mlp;

import java.util.ArrayList;

public class NeuralNetwork {

    private float output[];

    private ArrayList<Inputs> inputs;
    private int iterations;
    private Layear[] layears;
    
    public void setInputs(ArrayList<Inputs> inputs) {
        this.inputs = inputs;
    }

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
            this.output[i] = this.output[i] + bias[i];
        }

        return activationFunction(this.output);
    }

    public boolean train() {

        int count = 0;
        float[] y = new float[4];

        do {

            for(Inputs input : inputs) {
                boolean firstLoop = true;
                for(Layear layer : this.layears) {
                    if(!firstLoop) {
                        setOutputInInput(input);
                    }

                    this.output = run(input, layer.getWeights(), layer.getBias());

                    float[] auxOutput = {output[0], output[1], output[2], output[3]};
                    Inputs auxInput = new Inputs();
                    auxInput.x1 = input.x1;
                    auxInput.x2 = input.x2;
                    auxInput.x3 = input.x3;
                    auxInput.x4 = input.x4;

                    layer.setOutput(auxOutput);
                    layer.setInput(auxInput);

                    firstLoop = false;
                }

                for(int i = 0; i < this.output.length; i++ ) {
                    if(this.output[i] >= 0.5) {
                        y[i] = 1;
                    } else{
                        y[i] = 0;
                    }
                }

                if(validateResult(input, y)) {
                } else {
                    setOutputInInput(input);
                    backPropagation(input);
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

    private void backPropagation(Inputs input) {

        float learningRate = 0.1f;

        float[] error = {
            input.d1 - output[0],
            input.d2 - output[1],
            input.d3 - output[2],
        };


        float[] delta3 = {
            error[0] * output[0] * (1 - output[0]),
            error[1] * output[1] * (1 - output[1]),
            error[2] * output[2] * (1 - output[2]),
        };

            
        float[] delta2 = {
            sumDelta(delta3, this.layears[2].getWeights()[0]) * this.layears[2].getOutput()[0] * (1 - this.layears[2].getOutput()[0]),
            sumDelta(delta3, this.layears[2].getWeights()[1]) * this.layears[2].getOutput()[1] * (1 - this.layears[2].getOutput()[1]),
            sumDelta(delta3, this.layears[2].getWeights()[2]) * this.layears[2].getOutput()[2] * (1 - this.layears[2].getOutput()[2]),
            sumDelta(delta3, this.layears[2].getWeights()[3]) * this.layears[2].getOutput()[3] * (1 - this.layears[2].getOutput()[3]),
        };

        float[] delta1 = {
            sumDelta(delta2, this.layears[1].getWeights()[0]) * this.layears[1].getOutput()[0] * (1 - this.layears[1].getOutput()[0]),
            sumDelta(delta2, this.layears[1].getWeights()[1]) * this.layears[1].getOutput()[1] * (1 - this.layears[1].getOutput()[1]),
            sumDelta(delta2, this.layears[1].getWeights()[2]) * this.layears[1].getOutput()[2] * (1 - this.layears[1].getOutput()[2]),
            sumDelta(delta2, this.layears[1].getWeights()[3]) * this.layears[1].getOutput()[3] * (1 - this.layears[1].getOutput()[3]),
        };

        float[][] auxWeight3 = this.layears[3].getWeights();
        for(int i = 0; i < auxWeight3.length; i++) {
            for(int j = 0; j < delta3.length; j++) {
                auxWeight3[i][j] = auxWeight3[i][j] + (learningRate *  this.layears[3].getOutput()[i] * delta3[j]);
            }
        }

        float[][] auxWeight2 = this.layears[2].getWeights();
        for(int i = 0; i < this.layears[2].getWeights().length; i++) {
            for(int j = 0; j < delta2.length; j++) {
                auxWeight2[i][j] = auxWeight2[i][j] + learningRate *  this.layears[2].getOutput()[i] * delta2[j];
            }
        }

        float[][] auxWeight1 = this.layears[1].getWeights();
        for(int i = 0; i < this.layears[1].getWeights().length; i++) {
            for(int j = 0; j < delta1.length; j++) {
                auxWeight1[i][j] = auxWeight1[i][j] + learningRate *  this.layears[1].getOutput()[i] * delta1[j];
            }
        }

        float[] auxBias3 = this.layears[3].getBias();
        for(int i = 0; i < delta3.length; i++) {
            auxBias3[i] += learningRate * (-1 * this.layears[3].getBias()[i]) * delta3[i];
        }

        float[] auxBias2 = this.layears[2].getBias();
        for(int i = 0; i < delta2.length; i++) {
            auxBias2[i] += learningRate * (-1 * this.layears[2].getBias()[i]) * delta2[i];
        }

        float[] auxBias1 = this.layears[1].getBias();
        for(int i = 0; i < delta1.length; i++) {
            auxBias1[i] += learningRate * (-1 * this.layears[1].getBias()[i]) * delta1[i];
        }
    }

    private float sumDelta(float[] delta, float[] weights) {
        float result = 0;

        for(int i = 0; i < delta.length; i++) {
            result += weights[i] * delta[i];
        }
        return result;
    }

    public boolean test() {

        int acertos = 0;
        float[] y = new float[4];

        for(Inputs input : inputs) {
            for(Layear layer : this.layears) {

                this.output = run(input, layer.getWeights(), layer.getBias());
            }

            for(int i = 0; i < this.output.length; i++ ) {
                if(this.output[i] >= 0.5) {
                    y[i] = 1;
                } else{
                    y[i] = 0;
                }
            }

            if(validateResult(input, y)) {
                acertos++;
            } else {

            }
            System.out.println(y[0] + " " + y[1] + " " + y[2]);
        }

    System.out.println("Porcentagem de acertos: " + acertos/18);

        return true;
    }
}