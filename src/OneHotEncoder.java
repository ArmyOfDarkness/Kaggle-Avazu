/*
 * Used to manually one-hot-encode categorical features
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class OneHotEncoder {
	private static String input_train = "d:/Users/Brendan/Documents/Kaggle/Avazu/train_reduced.csv";
	private static String output_train = "d:/Users/Brendan/Documents/Kaggle/Avazu/train_reduced_onehot.csv";
	private static String input_test = "d:/Users/Brendan/Documents/Kaggle/Avazu/test_reduced.csv";
	private static String output_test = "d:/Users/Brendan/Documents/Kaggle/Avazu/test_reduced_onehot.csv";
	private static Map<String, ArrayList<String>> featureValues = new LinkedHashMap<>();
	private static Map<String, Integer> featureCounts = new HashMap<>();
	
	// keep feature only if it occurs at least a certain number of times;
	private static int threshold = 1000000;
	
	public static void main(String[] args) throws IOException {
		String line;
		Integer number;
		int totalFeatures = 0;
		
		// first pass through data creates lists of unique values for each feature
		// and keeps track of number of occurrences for each value
		int count = 0;
		BufferedReader train_reader = new BufferedReader(new FileReader(input_train));
		BufferedWriter train_writer = new BufferedWriter(new FileWriter(output_train));
		BufferedReader test_reader = new BufferedReader(new FileReader(input_test));
		BufferedWriter test_writer = new BufferedWriter(new FileWriter(output_test));
		String[] featureNames = train_reader.readLine().split(",");
		
		//set up arraylists in featureValues
		for (String s : featureNames) {
			if (s.equals("click")) continue;
			featureValues.put(s, new ArrayList<String>());
		}
		
		while ( (line = train_reader.readLine()) != null) {
			if (count++ % 1000000 == 0) System.out.println("Processed " + (count-1) + " rows");
			String[] data = line.split(",");
			for (int i = 1; i < data.length; i++) {
				if ( (number = featureCounts.get(featureNames[i]+"_"+data[i])) == null) {
					featureValues.get(featureNames[i]).add(data[i]);
					featureCounts.put(featureNames[i]+"_"+data[i], 1);
					totalFeatures++;
				} else {
					featureCounts.put(featureNames[i]+"_"+data[i], number+1);
				}
				
			}
		}
		train_reader.close();
		//print stats
		
		//create header row, eliminating features that don't pass the threshold count
		List<String> oneHotFeatures = new ArrayList<>();
		for (String key : featureValues.keySet()) {
			List<String> toRemove = new ArrayList<>();
			for (String value : featureValues.get(key)) {
				String combo = key + "_" + value;
				if (featureCounts.get(combo) > threshold) {
					oneHotFeatures.add(combo);
				} else {
					toRemove.add(value);
					System.out.println("Removed key: " + key + "; value: " + value);
					totalFeatures--;
				}
			}
			featureValues.get(key).removeAll(toRemove);
		}
		
		train_writer.write("click,");
		for (int i = 0; i < oneHotFeatures.size() - 1; i++) {
			train_writer.write(oneHotFeatures.get(i) + ",");
			test_writer.write(oneHotFeatures.get(i) + ",");
		}
		train_writer.write(oneHotFeatures.get(oneHotFeatures.size() - 1) + "\n");
		test_writer.write(oneHotFeatures.get(oneHotFeatures.size() - 1) + "\n");
		System.out.println("Total Features: " + totalFeatures);
		
		//second pass through data to encode features
		BufferedReader reader2 = new BufferedReader(new FileReader(input_train));
		reader2.readLine();
		count = 0;		
		
		while ( (line = reader2.readLine()) != null) {
			if (count++ % 1000000 == 0) System.out.println("Processed " + (count-1) + " rows");
			
			String[] train_data = line.split(",");
			List<Integer> train_outputs = new ArrayList<>();			
			
			for (int i = 1; i < featureNames.length; i++) {
				for (String value : featureValues.get(featureNames[i])) {
					train_outputs.add(train_data[i].equals(value) ? 1 : 0);
				}
			}
			
			train_writer.write(train_data[0] + ","); //record click for training data
			for (int i = 0; i < train_outputs.size() - 1; i++) {
				train_writer.write(train_outputs.get(i) + ",");
			}
			train_writer.write(train_outputs.get(train_outputs.size()-1) + "\n");
		}
		reader2.close();
		train_writer.close();
		
		while ( (line = test_reader.readLine()) != null) {
			if (count++ % 1000000 == 0) System.out.println("Processed " + (count-1) + " rows");
			
			String[] test_data = line.split(",");
			List<Integer> test_outputs = new ArrayList<>();
			
			for (int i = 1; i < featureNames.length; i++) {
				for (String value : featureValues.get(featureNames[i])) {
					test_outputs.add(test_data[i-1].equals(value) ? 1 : 0);
				}
			}
			
			for (int i = 0; i < test_outputs.size() - 1; i++) {
				test_writer.write(test_outputs.get(i) + ",");
			}
			test_writer.write(test_outputs.get(test_outputs.size()-1) + "\n");
		}
		test_reader.close();
		test_writer.close();
	}

}
