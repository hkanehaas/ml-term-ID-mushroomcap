import java.io.*;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.lang.Math;

public class MainClass {
	public static void main(String[] args) throws FileNotFoundException {
		//Initialize weights
		double[] weights = new double[11];
		int N = 6499;
		int NTest = 1625;
		int wsize = weights.length;
		double lr = 0.5; //learning rate
		
		//importing CSV
		ArrayList<ArrayList<Integer>> xdata = featuresCSV(N, wsize, "\\src\\java-train-features.csv");
		ArrayList<Integer> ydata = readLabelsCSV(N, "\\src\\java-train-targets.csv"); 
		ArrayList<Integer> yhat = new ArrayList<Integer>(6499);
		for (int i = 0; i < N; i++) {
			yhat.add(7); //setting up yhat for later input	
		}
		//Test CSVs
		ArrayList<ArrayList<Integer>> xdataTest = featuresCSV(NTest, wsize, "\\src\\java-test-features.csv");
		ArrayList<Integer> ydatatest = readLabelsCSV(NTest, "\\src\\java-test-targets.csv"); 
		ArrayList<Integer> yhatTest = new ArrayList<Integer>(NTest);
		for (int i = 0; i < NTest; i++) {
			yhatTest.add(7); //setting up yhat for later input	
		}		
		
		weights = gradDesc(200, wsize, N, lr, ydata, xdata);
		yhat = logReg(11, 6499, weights, xdata, yhat, false);
		System.out.println("\n-------accuracy of training data----------");
		accuracy(N, ydata, yhat);
		
		//Test weights on test data
		yhatTest = logReg(11, NTest, weights, xdataTest, yhatTest, false);
		System.out.println("\n-------accuracy of test data----------");
		accuracy(NTest, ydatatest, yhatTest);
		
		userSelection(weights);
		
		
		
	} //end main method	
	
	static double[] gradDesc(int epochCount, int weightCount, int observationCount, double learningRate,
							 ArrayList<Integer> ydata,  ArrayList<ArrayList<Integer>> xdata) {
		Random r = new Random(11172017);
		double[] weights = r.doubles(11, 0.5,2).toArray();
		weights[0] = 1.0; //bias
		double[] numerator = new double[weightCount];
		double[] denominator = new double[observationCount];
		double[] sum = new double[weightCount];
		double temp1 = 0;
		double[]  ynwTx = new double[weightCount];
		int j = 0;
		double lr = learningRate;
		
		//print weights
		System.out.println("Random Starting Weights: " + Arrays.toString(weights));
		
		//Calculating gradient descent	
		for (int k = 0; k < epochCount; k++) { 
		for (int i = 0; i < observationCount; i++) { //loops through entire data set.
			
			//summation
			for (j = 0; j < weightCount; j++) { //summation
				numerator[j] = ydata.get(i) * xdata.get(i).get(j); //yn*xn
				ynwTx[j] = ydata.get(i) * ( weights[j] * xdata.get(i).get(j)); //ynwTx for e^ynwTx
				temp1 = temp1 + ynwTx[j];
			} //end summation
			
			denominator[i] =  1 + Math.exp(-1 * temp1); // 1 + e^ynwTx
			temp1 = 0; //reset for next run through the loop
			for ( j = 0; j < weightCount; j++) { //summation	
				sum[j] = numerator[j]/denominator[i] + sum[j];
			}
			
		} // end epoch
		
		//multiply by -1/N and learning rate.
		for ( j = 0; j < weightCount; j++) { //summation
			
			sum[j] = (sum[j] * lr * -1) / observationCount ; //vt <<-- this is zero-ing out on sums 1, 2, 4, 5, 7, 8, 9 
			weights[j] = weights[j] + sum[j]; // w = w + gradient descent
			sum[j] = 0.0; // reset summation to zero for next epoch
		}

		} // run epochCount epochs
		//print weights:
		System.out.println("New Weights: " + Arrays.toString(weights));
		
		return weights;
	
	}
	
	static ArrayList logReg(int weightCount, int observationCount, double[] weights,
							ArrayList<ArrayList<Integer>> xdata, ArrayList<Integer> yhat, boolean includProb) {
				
		int i = 0;
		int j = 0;
		double temp0 = 0;
		double[]  wTx = new double[weightCount];
		double[] probsum = new double[observationCount]; //probability summation
	    
	  //prediction using logistic regression (yhat) output as probability
  		for (i = 0; i < observationCount; i++) { //loops through entire data set.
 
  			//summation
  			for ( j = 0; j < weightCount; j++) { //summation
  			
  			wTx[j] =  weights[j] * xdata.get(i).get(j); //what exponent will be raised to in denominator of logistic function
  			temp0 = temp0 + wTx[j]; //summation of prev line
  			}
  			probsum[i] = 1 / (1 + Math.exp(-1 * temp0)); //yHat probabilities.
  			
  			if (includProb) {
  				
  				double probOut = probsum[i] * 100;
  				System.out.println("\nAfter analysis we are: " +  String.format("%.2f%%", probOut) + " certain your mushroom is edible");
  			}
  			
  			temp0 = 0;
  			
  			//Threshold populate yhat array
  			if (probsum[i] >= 0.75) {
  				yhat.set(i, 0); //Edible
  			}
  			else { yhat.set(i, 1);} //Poison
  		}
		return yhat;
		
	  }
	
	static double accuracy(int observationCount, ArrayList<Integer> ydata, ArrayList<Integer> yhat) {
		int i = 0;
		double accuracy = 0;
		for (i = 0; i < observationCount; i++) { //loops through entire data set.
			accuracy = (ydata.get(i) == -1) ? Math.abs(0 - yhat.get(i)) + accuracy : Math.abs(ydata.get(i) - yhat.get(i)) + accuracy;
		}
		accuracy = 1 - (accuracy/observationCount);
		accuracy = accuracy * 100;
  		System.out.println("accuracy: " +  String.format("%.2f%%", accuracy));
		return accuracy;
	}
	
	static ArrayList<Integer>  readLabelsCSV(int observationCount, String fileName ) throws FileNotFoundException {
		
		ArrayList<Integer> labels = new ArrayList<Integer>(observationCount);
		String dir = System.getProperty("user.dir");
		Scanner labelsSC = new Scanner(new File( dir + fileName));
		int i = 0;

		while (labelsSC.hasNext())
		{  	
			String newlabel = labelsSC.nextLine();
			String labelVals[] = newlabel.split(",");
			int nexty = Integer.parseInt(labelVals[0]);
			//input y/target matrix
			labels.add(nexty);		
			i++;
		}   
		labelsSC.close();
		
		return labels;
	}
	
	static ArrayList<ArrayList<Integer>> featuresCSV(int observationCount, int featuresCount, String fileName ) throws FileNotFoundException {
		//importing CSV
		ArrayList<ArrayList<Integer>> features = new ArrayList<>(observationCount);		
		String dir = System.getProperty("user.dir");	
		Scanner featuresSC = new Scanner(new File( dir + fileName));
		int i = 0;
		int j = 0;
		while (featuresSC.hasNext())
		{  
			j = 0;		
			String newLine = featuresSC.nextLine();
			String xvals[] = newLine.split(",");

			features.add(new ArrayList());
			features.get(i).add(1); //bias x0
			//Input each line of the x/features matrix
			for (j = 0; j < featuresCount - 1; j++) {  
				
				int xval = Integer.parseInt(xvals[j]);
				features.get(i).add(xval);
				}	
			i++;
		}   
		featuresSC.close();

		return features;
		
		
	}
	
	public static void userSelection(double[] weights) {
		
		double[] weightsIn = weights;
		int wsize = weightsIn.length;
		ArrayList<ArrayList<Integer>> userObservation = new ArrayList<>(1);
		userObservation.add(new ArrayList(1));
		userObservation.get(0).add(1); //bias x0
		
		ArrayList<Integer> yhat = new ArrayList<Integer>(1);
		yhat.add(7);
		String selection = "y";
		
		System.out.println("\n-------Test a single mushroom----------");		
		
		Scanner userInput = new Scanner(System.in);
		System.out.println("Would you like to test a mushroom? type y for yes, all others exit.");
		selection = userInput.nextLine();
		
		while (selection.equalsIgnoreCase("y")) {
			
			//Feature: bruised
			System.out.println("Is the mushroom bruised? Y/N");
			selection = userInput.nextLine();
				if (selection.equalsIgnoreCase("y")) {
					userObservation.get(0).add(1);
				}
				else {userObservation.get(0).add(0);}
			
			//Feature: gill-attachment
			System.out.println("Is the gill attached? Y/N");
			selection = userInput.nextLine();
				if (selection.equalsIgnoreCase("y")) {
					userObservation.get(0).add(1);
				}
				else {userObservation.get(0).add(0);}
				
			//Feature: gill-spacing
			System.out.println("Is the gill spaced crowded? Y/N");
			selection = userInput.nextLine();
				if (selection.equalsIgnoreCase("y")) {
					userObservation.get(0).add(0);
				}
				else {userObservation.get(0).add(1);}
			
			//Feature: gill-size
			System.out.println("Is the gill size broad? Y/N");
			selection = userInput.nextLine();
				if (selection.equalsIgnoreCase("y")) {
					userObservation.get(0).add(1);
				}
				else {userObservation.get(0).add(0);}
				
			//Feature: stalk-surface-above-ring
			System.out.println("Is the stalk surface ABOVE the ring Fibrous, Scaly, Silky, or Smooth?");
			selection = userInput.nextLine();
			if (selection.equalsIgnoreCase("Fibrous")) {
				userObservation.get(0).add(1);
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
			}
			else if  (selection.equalsIgnoreCase("Scaly")){
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
			}
			else if  (selection.equalsIgnoreCase("Silky")){
				userObservation.get(0).add(0);
				userObservation.get(0).add(1);
				userObservation.get(0).add(0);
			}
			else { //Smooth
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
				userObservation.get(0).add(1);
			}
			
			//Feature: stalk-surface-below-ring
			System.out.println("Is the stalk surface BELOW the ring Fibrous, Scaly, Silky, or Smooth?");
			selection = userInput.nextLine();
			if (selection.equalsIgnoreCase("Fibrous")) {
				userObservation.get(0).add(1);
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
			}
			else if  (selection.equalsIgnoreCase("Scaly")){
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
			}
			else if  (selection.equalsIgnoreCase("Silky")){
				userObservation.get(0).add(0);
				userObservation.get(0).add(1);
				userObservation.get(0).add(0);
			}
			else { //Smooth
				userObservation.get(0).add(0);
				userObservation.get(0).add(0);
				userObservation.get(0).add(1);
			}
			
			Boolean prob = true;
			
			yhat = logReg(wsize, 1, weightsIn, userObservation, yhat, prob);
			
			if (yhat.get(0) == 0) {
				System.out.println("\nBased on perviously tested data this mushroom might be edible");
			}
			else {System.out.println("\nBased on perviously tested data this mushroom it likely to be >>>POISONIOUS<<<");}
			
			
			System.out.println("\nWould you like to test another? type y for yes, all others exit.");
			selection = userInput.nextLine();
		}
		
		System.out.println("\n-------Exited----------");	
		userInput.close();

    }
	
} //end MainClass


