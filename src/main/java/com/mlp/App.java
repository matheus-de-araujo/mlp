package com.mlp;

import java.io.*;
import java.util.ArrayList;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args ) throws NumberFormatException, IOException
    {
        ArrayList<Inputs> inputs = getinputs(); 

        NeuralNetwork network = new NeuralNetwork(inputs, 1000);
        network.train();
    }

    public static ArrayList<Inputs> getinputs() throws NumberFormatException, IOException {
        FileInputStream fileInputStream = new FileInputStream("src/dado.csv");
        BufferedReader BufferedReader = new BufferedReader(new InputStreamReader(fileInputStream));
        ArrayList<Inputs> inputs = new ArrayList<>();

        String strLine;
        while((strLine = BufferedReader.readLine()) != null) {
            Inputs input = new Inputs();
            String[] data = strLine.split(";");
            input.x1 = Float.parseFloat(data[0]);
            input.x2 = Float.parseFloat(data[1]);
            input.x3 = Float.parseFloat(data[2]);
            input.x4 = Float.parseFloat(data[3]);
            input.d1 = Float.parseFloat(data[4]);
            input.d2 = Float.parseFloat(data[5]);
            input.d3 = Float.parseFloat(data[6]);

            inputs.add(input);
        }

        BufferedReader.close();
        return inputs;
    }
}
