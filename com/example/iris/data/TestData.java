/**
 * 
 */
package com.example.iris.data;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import org.apache.mahout.classifier.sgd.AdaptiveLogisticRegression;
import org.apache.mahout.classifier.sgd.CrossFoldLearner;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.ModelSerializer;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.Dictionary;

import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;

/**
 * @author himanshusharma
 *
 */
public class TestData {
	List<Integer> train;
	List<Integer> test;
	ArrayList<Integer> target;
	List<Vector> data;
	ArrayList<Integer> order;
	List<String> raw = new ArrayList<String>();
	Random random;

	public void buildTrainData() throws IOException {

		// Snip ...
		RandomUtils.useTestSeed();
		Splitter onComma = Splitter.on(",");

		String csvFile = "/Users/himanshusharma/Downloads/train-final.csv";
		BufferedReader br = null;
		String lines = "";

		try {

			br = new BufferedReader(new FileReader(csvFile));
			while ((lines = br.readLine()) != null) {
				raw.add(lines);

			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		// System.out.println(raw_train);
		// holds features
		data = Lists.newArrayList();

		// holds target variable
		target = Lists.newArrayList();

		// for decoding target values
		Dictionary dict = new Dictionary();

		// for permuting data later
		order = Lists.newArrayList();
		for (String line : raw.subList(1, raw.size())) {
			// ; gets a list of indexes
			order.add(order.size());

			// parse the predictor variables
			DenseVector v = new DenseVector(41);
			DenseVector v_withOriginalIntercept = new DenseVector(40);

			v.set(0, 1);
			int i = 1;
			Iterable<String> values = onComma.split(line);

			for (String value : Iterables.limit(values, 2)) {
				value = value.replaceAll("\"", "");
				v.set(i++, Double.parseDouble(value));

			}
			v_withOriginalIntercept = (DenseVector) v.normalize();
			v_withOriginalIntercept.set(0, 1);
			data.add(v);

			// and the target
			target.add(dict.intern(Iterables.get(values, 40)));
		}

	}

	public static void main(String[] args) throws Exception {
		iris t1 = new iris();
		t1.buildTrainData();
		t1.trainable();

	}

	public void trainable() throws Exception {

		double heldOutPercentage = 0.10;
		int cutoff = (int) (heldOutPercentage * order.size());
		Collections.shuffle(order);
		List<Integer> test = order.subList(0, cutoff);
		List<Integer> train = order.subList(cutoff, order.size());

		double accuracy = 0.0, temp = 0.0;

		AdaptiveLogisticRegression lr = new AdaptiveLogisticRegression(40, 41,
				new L1());
		for (Integer k : train) {
			lr.train(target.get(k), data.get(k));
		}
		lr.close();
		ModelSerializer.writeBinary(
				"/Users/himanshusharma/Downloads/alr2.model", lr.getBest()
						.getPayload().getLearner());// .getModels().get(0));
		InputStream in = new FileInputStream(
				"/Users/himanshusharma/Downloads/alr2.model");
		CrossFoldLearner best = ModelSerializer.readBinary(in,
				CrossFoldLearner.class);
		System.out.println("auc=" + best.auc() + "percentCorrect="
				+ best.percentCorrect() + "LogLikelihood="
				+ best.getLogLikelihood());

		in.close();
		// check the accuracy on held out data
		int x = 0, r = 0;
		for (Integer k : test) {
			// Test testProfile;

			r = best.classifyFull(data.get(k)).maxValueIndex();
			// System.out.println("r="+r+" "+"target_train.get(k)="+target_train.get(k));
			x += r == target.get(k) ? 1 : 0;
		}

		accuracy = (double) (x) / (double) (test.size());
		accuracy *= 100.0;

		System.out.printf("accuracy is %f\n", accuracy);
	}
}
