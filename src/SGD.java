/* 
 * Originally written in Python by tinrtgu
 * Posted to the Avazu CTR prediction competition on Kaggle
 * http://www.kaggle.com/c/avazu-ctr-prediction/forums/t/10927/beat-the-benchmark-with-less-than-1mb-of-memory
 * 
 * Modified by Brendan Borin:
 * translated from Python to Java for 10x increase in speed
 * added optional cross validation capability
 * added optional inclusion of GBT predictions as input features
 */



import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SGD {
	private static String train = "d:/Users/Brendan/Documents/Kaggle/Avazu/train.csv";
	private static String train2 = "d:/Users/Brendan/Documents/Kaggle/Avazu/train_H2O_trees3.csv";
	private static String test = "d:/Users/Brendan/Documents/Kaggle/Avazu/test.csv";
	private static String test2 = "d:/Users/Brendan/Documents/Kaggle/Avazu/test_H2O_trees3.csv";
	private static String predict = "d:/Users/Brendan/Documents/Kaggle/Avazu/submission50.csv";
	private static String stats = "d:/Users/Brendan/Documents/Kaggle/Avazu/LogLossStatstrees10.csv";
	
	private static double alpha = .1;
	private static double beta = .1;
	private static double L1 = 1;
	private static double L2 = 1;
	
	private static int featureSize = (int) Math.pow(2, 26);
	private static boolean interactions = true;
	private static boolean includeOriginals = true;
	private static boolean includeTrees = false;
	private static boolean includeSelectInteractions = false;
	
	private static int iterations = 1;
	private static int holdAfter = 0;
	private static int holdOut = 0;
	private static int Kfold = 1;
	
	//n = squared sum of past gradients
	//z = weights
	//w = lazy weights
	private static double[] n; 
	private static double[] z;
	private static Map<Integer, Double> w;
	
	static void update(List<Integer> x, double p, int y) {
		// gradient under logloss
        double g = p - y;
        
        // update z and n
        for (int i : x) {
            double sigma = ((Math.sqrt(n[i] + g * g) - Math.sqrt(n[i])) / alpha);
            z[i] += g - sigma * w.get(i);
            n[i] += g * g;
        }
	}
	
	static double logLoss(double p, int y) {
		double p_adj = Math.max(Math.min(p, 1 - 10e-15), 10e-15);
		return (y == 1 ? -Math.log(p_adj) : -Math.log(1-p_adj));
	}
	
	static double prediction(List<Integer> x) {
		double wTx = 0;
		Map<Integer, Double> w_local = new HashMap<>();
		
		for (int i : x) {
			//System.out.print(z[i] + " ");
			int sign = (z[i] < 0 ? -1 : 1);
			if (sign * z[i] <= L1) {
				w_local.put(i, 0d);
			}
			else {
				w_local.put(i, (double) ((sign * L1 - z[i]) / ((beta + Math.sqrt(n[i])) / alpha + L2)));
			}
			//System.out.println("values: " + w_local.values());
			wTx += w_local.get(i);
		}
		//System.out.print("\n");
		
		// cache the current w for update stage
        w = w_local;

        // bounded sigmoid function, this is the probability estimation
        return (1 / (1 + Math.exp(-Math.max(Math.min(wTx, 35.), -35.))));
	}
	
	public static void main(String[] args) throws IOException {
		String line = "";

		//get time
		long startTime = System.currentTimeMillis();
		
		BufferedWriter loglossstats = new BufferedWriter(new FileWriter(stats));
				
		for (int k = 1; k <= Kfold; k++) {
			n = new double[featureSize]; 
			z = new double[featureSize];
			w = new HashMap<>();
			//alpha = alpha_0;
			
			for (int i = 1; i <= iterations; i++) {
				BufferedReader trainReader = new BufferedReader(new FileReader(train));
				BufferedReader trainReader2 = new BufferedReader(new FileReader(train2));
				String[] featureNames = trainReader.readLine().split(",");
				String[] featureNames2 = trainReader2.readLine().split(",");
				int t = 0;
				double loss = 0;
				int count = 0;
				
				while ( (line = trainReader.readLine()) != null ) {
					//if (count % 5000000 == 0) System.out.println("Processed " + count + " lines.");
					String[] data = line.split(",");
					String[] data2 = trainReader2.readLine().split(",");
					t++;
					
					int date = Integer.parseInt(data[2].substring(4,6));
					int hour = Integer.parseInt(data[2].substring(data[2].length() - 2));
					
					//remove data from beginning
					//if (date < 23) continue;					
					
					data[2] = data[2].substring(data[2].length()-2);
					int y = Integer.parseInt(data[1]);
					List<Integer> x = new ArrayList<>();					
					
					//hash original features
					if (includeOriginals) {
						for (int j = 2; j < data.length; j++) {
						
						//combine device_type, device_conn_type and remove originals
						if (j == 14) data[j] = data[j] + "_" + data[j+1];
						if (j == 15) continue;
						
						int index = Math.abs((featureNames[j] + "_" + data[j]).hashCode()) % featureSize;
						x.add(index);
						}
					}
					
					//hash feature interactions
					if (interactions) {
						int len = x.size();
						for (int m = 2; m < len; m++) {
							for (int n = m+1; n <= len; n++) {
								int index = Math.abs((Integer.toString(x.get(m)) + "_" + Integer.toString(x.get(n))).hashCode()) % featureSize;
								x.add(index);
							}
						}
					}
										
					//hash gradient boosted tree features
					if (includeTrees) {
						for (int j = 0; j < data2.length; j++) {
							int index = Math.abs((featureNames2[j] + "_" + data2[j]).hashCode()) % featureSize;
							x.add(index);
						}
					}
										
					//select interactions
					if (includeSelectInteractions) {
						int[] selectFeatures = {5,6,7,10,16,18,20,23};
						for (int m = 0; m < selectFeatures.length - 1; m++) {
							int m_index = selectFeatures[m];
							for (int n = m+1; n < selectFeatures.length; n++) {
								int n_index = selectFeatures[n];
								int index = Math.abs( (featureNames[m_index] + "_" + data[m_index] + "_" + featureNames[n_index] + "_" + data[n_index]).hashCode()) % featureSize;
								x.add(index);
								
								for (int o = n+1; o < selectFeatures.length; o++) {
									int o_index = selectFeatures[o];
									int index2 = Math.abs( (featureNames[m_index] + "_" + data[m_index] + "_" + featureNames[n_index] + "_" + data[n_index] + "_" + featureNames[o_index] + "_" + data[o_index]).hashCode()) % featureSize;
									x.add(index2);
								}
							}
						}
					}
										
					
					
					//get prediction
					double p = prediction(x);
					/*
					if ((holdAfter > 0 && date > holdAfter) || (holdOut > 0 && t % holdOut == 0) || (Kfold > 1 && t % Kfold == k-1)) {
						loss += logLoss(p, y);
						count++;
					}
					else {
						update(x, p, y);
					}*/
					
					
					double logloss = logLoss(p, y);
					loss += logloss;
					count++;
					//if (i != iterations) {
						update(x, p, y);
					//}
					
					
					
					//plot performance metrics
					if (k == 1 && t % 10000 == 0) {
						//System.out.println("Loss after " + t + ": " + loss/count);
						loglossstats.write(hour + "," + p + "," + loss + "," + loss/count + "\n");
					}
				}
				
				if (count == 0) count = 1;
				long time = System.currentTimeMillis() - startTime;
				int sec = (int)(time/ 1000) % 60 ;
				int min = (int)((time/ (1000*60)) % 60);
				int hr = (int)((time/ (1000*60*60)) % 24);
				String timeString = String.format("%02d:%02d:%02d", hr, min, sec);
				System.out.println("Fold " + k + "; Iteration " + i + "; finished, validation logloss: " + loss/count + ", elapsed time: " + timeString);
				trainReader.close();
				trainReader2.close();
			}
		}
		
		loglossstats.close();
		
		if (Kfold == 1) {
			try (BufferedReader testReader = new BufferedReader(new FileReader(test));
					BufferedReader testReader2 = new BufferedReader(new FileReader(test2));
					BufferedWriter predWriter = new BufferedWriter(new FileWriter(predict))) {
				String[] featureNames = testReader.readLine().split(",");
				String[] featureNames2 = testReader2.readLine().split(",");
			
				predWriter.write("id,click\n");
			
				while ( (line = testReader.readLine()) != null ) {
					String[] data = line.split(",");
					String[] data2 = testReader2.readLine().split(",");
				
					data[1] = data[1].substring(data[1].length() - 2);
					List<Integer> x = new ArrayList<>();
					
					//hash original features
					if (includeOriginals) {
						for (int j = 1; j < data.length; j++) {
							
							//combine device_type, device_conn_type and remove originals
							if (j == 13) data[j] = data[j] + "_" + data[j+1];
							if (j == 14) continue;
							
							int index = Math.abs((featureNames[j] + "_" + data[j]).hashCode()) % featureSize;
							x.add(index);
						}
					}
					
					if (interactions) {
						int len = x.size();
						for (int m = 1; m < len; m++) {
							for (int n = m+1; n <= len; n++) {
								int index = Math.abs((Integer.toString(x.get(m)) + "_" + Integer.toString(x.get(n))).hashCode()) % featureSize;
								x.add(index);
							}
						}
					}
										
					//hash gradient boosted tree features
					if (includeTrees) {
						for (int j = 0; j < data2.length; j++) {
							int index = Math.abs((featureNames2[j] + "_" + data2[j]).hashCode()) % featureSize;
							x.add(index);
						}
					}
										
					//select interactions
					if (includeSelectInteractions) {
						int[] selectFeatures = {4,5,6,9,15,17,19,22};
						for (int m = 0; m < selectFeatures.length - 1; m++) {
							int m_index = selectFeatures[m];
							for (int n = m+1; n < selectFeatures.length; n++) {
								int n_index = selectFeatures[n];
								int index = Math.abs( (featureNames[m_index] + "_" + data[m_index] + "_" + featureNames[n_index] + "_" + data[n_index]).hashCode()) % featureSize;
								x.add(index);
								
								for (int o = n+1; o < selectFeatures.length; o++) {
									int o_index = selectFeatures[o];
									int index2 = Math.abs( (featureNames[m_index] + "_" + data[m_index] + "_" + featureNames[n_index] + "_" + data[n_index] + "_" + featureNames[o_index] + "_" + data[o_index]).hashCode()) % featureSize;
									x.add(index2);
								}
							}
						}
					}
														
					//get prediction
					double p = prediction(x);
					predWriter.write(data[0] + "," + Double.toString(p) + "\n");
				}
				testReader.close();
				predWriter.close();
			}
		}
	}
}
