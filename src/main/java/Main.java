import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.lazy.IBk;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;

import java.io.*;

public class Main {

    public static void main(String[] args) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File("test.csv"));
        Instances testSet = loader.getDataSet();
        loader.setSource(new File("train.csv"));
        Instances trainSet = loader.getDataSet();
        if (trainSet.classIndex() == -1) {
            trainSet.setClassIndex(trainSet.numAttributes() - 1);
        }
        if (testSet.classIndex() == -1) {
            testSet.setClassIndex(testSet.numAttributes() - 1);
        }
        predict(testSet, trainSet);
        RandomForest rfTree = new RandomForest();
        Evaluation eval = new Evaluation(trainSet);
      /*  Evaluation eval = new Evaluation(trainSet);
        int bestTreesNumber = 10;
        double bestQualityMeasure = 0;
        for (int treesNumber = 10; treesNumber < 200; treesNumber +=10) {
                rfTree.setNumTrees(200);
                rfTree.buildClassifier(trainSet);
                eval.crossValidateModel(rfTree, trainSet, 10, new Random(1));
                double qualityMeasure = eval.correct()/ (eval.correct() + eval.incorrect());
                System.out.println(qualityMeasure);
                if (qualityMeasure > bestQualityMeasure) {
                    bestQualityMeasure = qualityMeasure;
                    bestTreesNumber = treesNumber;
                }
        }
        System.out.println("Best trees number for RF: " + bestTreesNumber); */
        rfTree.setNumTrees(100);
        IBk iBk = new IBk();
        iBk.setKNN(200);
       /* int bestNearestNeighboursNumber = 10;
        double bestQualityMeasure = 0;
        for (int nearestNeighbourNumber = 10; nearestNeighbourNumber < 200; nearestNeighbourNumber+=10) {
            iBk.setKNN(nearestNeighbourNumber);
            iBk.buildClassifier(trainSet);
            eval.crossValidateModel(iBk, trainSet, 10, new Random(1));
            double qualityMeasure = eval.correct()/ (eval.correct() + eval.incorrect());
            System.out.println(qualityMeasure);
            if (qualityMeasure > bestQualityMeasure) {
                bestQualityMeasure = qualityMeasure;
                bestNearestNeighboursNumber = nearestNeighbourNumber;
            }
        }
        System.out.println("Best nearest neighbours number for iBk: " + bestNearestNeighboursNumber);*/
        MultilayerPerceptron perceptron = new MultilayerPerceptron();
        perceptron.setTrainingTime(50);
        perceptron.setLearningRate(0.01);
        /*double bestQualityMeasure = 0;
        double bestLearningRate = 0.01;
        int bestEpochs = 50;
        for (double learningRate = 0.01; learningRate <=0.2; learningRate +=0.1) {
            for (int epochs = 50; epochs < 300; epochs +=10) {
                perceptron.setLearningRate(learningRate);
                perceptron.setTrainingTime(epochs);
                perceptron.buildClassifier(trainSet);
                eval.crossValidateModel(perceptron, trainSet, 10, new Random(1));
                double qualityMeasure = eval.correct() / (eval.correct() + eval.incorrect());
                System.out.println("Q: " + qualityMeasure + ", rate: " + learningRate + ", epochs: " + epochs);
                if (qualityMeasure > bestQualityMeasure) {
                    bestQualityMeasure = qualityMeasure;
                    bestLearningRate = learningRate;
                    bestEpochs = epochs;
                }
            }
        }*/
      /*  CfsSubsetEval eval2 = new CfsSubsetEval();
        GreedyStepwise search = new GreedyStepwise();
        search.setSearchBackwards(true);
        AttributeSelection attributeSelection = new AttributeSelection();
        attributeSelection.setEvaluator(eval2);
        attributeSelection.setSearch(search);
        attributeSelection.setRanking(true);
        attributeSelection.SelectAttributes(trainSet);
        Instances trainSet2 = attributeSelection.reduceDimensionality(trainSet);

        ASEvaluation eval3 = new GainRatioAttributeEval();
        ASSearch search2 = new Ranker();
        attributeSelection.setEvaluator(eval3);
        attributeSelection.setSearch(search2);
        attributeSelection.SelectAttributes(trainSet);
        Instances trainSet3 = attributeSelection.reduceDimensionality(trainSet);

        eval.crossValidateModel(rfTree, trainSet, 10, new Random(1));
        printQuality(eval);
        eval.crossValidateModel(rfTree, trainSet2, 10, new Random(1));
        printQuality(eval);
        eval.crossValidateModel(rfTree, trainSet3, 10, new Random(1));
        printQuality(eval);

        eval.crossValidateModel(perceptron, trainSet, 10, new Random(1));
        printQuality(eval);
        eval.crossValidateModel(perceptron, trainSet2, 10, new Random(1));
        printQuality(eval);
        eval.crossValidateModel(perceptron, trainSet3, 10, new Random(1));
        printQuality(eval); */
    }

    private static void printQuality(Evaluation eval) {
        double qualityMeasure = eval.correct() / (eval.correct() + eval.incorrect());
        System.out.println(qualityMeasure);
    }

    private static void predict(Instances testSet, Instances trainSet) throws Exception {
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(new File("prediction.csv"))));
        RandomForest randomForest = new RandomForest();
        pw.println("id,class");
        randomForest.setNumTrees(100);
        randomForest.buildClassifier(trainSet);
        for (int i = 0; i < testSet.numInstances() ; i++) {
            int index = (int) randomForest.classifyInstance(testSet.instance(i));
            String currentClass = trainSet.classAttribute().value(index);
            int id = i * 2 + 1;
            pw.println(id + "," + currentClass);
        }
        pw.close();
    }
}
