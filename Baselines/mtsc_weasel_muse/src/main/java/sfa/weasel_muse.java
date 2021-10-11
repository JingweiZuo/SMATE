// Copyright (c) 2017 - Patrick Schäfer (patrick.schaefer@hu-berlin.de)
// Distributed under the GLP 3.0 (See accompanying file LICENSE)

package sfa;

import java.io.File;
import java.io.IOException;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import sfa.classification.Classifier;
import sfa.classification.MUSEClassifier;
import sfa.timeseries.MultiVariateTimeSeries;
import sfa.timeseries.TimeSeries;
import sfa.timeseries.TimeSeriesLoader;

@RunWith(JUnit4.class)
public class weasel_muse {

    // The multivariate datasets to use
    public static String[] datasets = new String[]{
            "DigitShapeRandom",
            "PenDigits"
    };


    @Test
    public void testMultiVariatelassification() throws IOException {
        try {
            // the relative path to the datasets
            ClassLoader classLoader = SFAWordsTest.class.getClassLoader();

            File dir = new File(classLoader.getResource("datasets/multivariate/").getFile());
            //File dir = new File("/Users/bzcschae/workspace/similarity/datasets/multivariate/");

            TimeSeries.APPLY_Z_NORM = false;

            for (String s : datasets) {
                File d = new File(dir.getAbsolutePath() + "/" + s);
                if (d.exists() && d.isDirectory()) {
                    for (File train : d.listFiles()) {
                        if (train.getName().toUpperCase().endsWith("TRAIN3")) {
                            File test = new File(train.getAbsolutePath().replaceFirst("TRAIN3", "TEST3"));

                            if (!test.exists()) {
                                System.err.println("File " + test.getName() + " does not exist");
                                test = null;
                            }

                            Classifier.DEBUG = false;

                            boolean useDerivatives = true;
                            MultiVariateTimeSeries[] trainSamples = TimeSeriesLoader.loadMultivariateDatset(train, useDerivatives);
                            MultiVariateTimeSeries[] testSamples = TimeSeriesLoader.loadMultivariateDatset(test, useDerivatives);

                            MUSEClassifier muse = new MUSEClassifier();
                            MUSEClassifier.Score museScore = muse.eval(trainSamples, testSamples);
                            System.out.println("\nDataset; Method; Training_accuracy; Testing_accuracy");
                            System.out.println(s + ";" + museScore.toString()+"\n");
                        }
                    }
                } else{
                    // not really an error. just a hint:
                    System.out.println("Dataset could not be found: " + d.getAbsolutePath() + ".");
                }
            }
        } finally {
            TimeSeries.APPLY_Z_NORM = true; // FIXME static variable breaks some test cases!
        }
    }
}