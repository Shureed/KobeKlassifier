import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.meta.ClassificationViaClustering;
import weka.classifiers.meta.ClassificationViaRegression;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.clusterers.*;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.sql.Date;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by shureedkabir on 4/24/16.
 */
public class KobeKlassifier {
    public static void main(String[] args) throws Exception {
//        use the following lines instead if using raw CSV data that has not been pre processed
//        CSVLoader loader = new CSVLoader();
//        loader.setFile(new File("data/kobetraindata.csv"));
//        Instances instances = preprocess(loader.getDataSet());
//        instances = convertDateFormat(instances);


        ArffLoader loader = new ArffLoader();
        loader.setFile(new File("data/kobedata.arff"));
        Instances instances = loader.getDataSet();

        instances.setClassIndex(instances.attribute("shot_made_flag").index()); //set class index to shot made flag

        classificationTest(instances);
//        clusterTest(instances);
    }

    public static void classificationTest(Instances instances) throws Exception {
        Classifier classifier;
        classifier = new DecisionTable();
//        classifier = new J48();
//        classifier = new NaiveBayes();
//        classifier = new SimpleLogistic();
//        classifier = new ClassificationViaClustering();
//        ((ClassificationViaClustering)classifier).setClusterer(new MakeDensityBasedClusterer());
//        ((ClassificationViaClustering)classifier).setClusterer(new SimpleKMeans());
//        classifier = new EnsembleClassifier();

        Evaluation evaluation = new Evaluation(instances);
        int folds = 10;
        evaluation.crossValidateModel(classifier, instances, folds, new Random(0));

        System.out.println(evaluation.toSummaryString());
    }

    public static void removeInstancesWithMissing(Instances instances){
//        used because ClassificationViaClustering cannot handle instances with missing values
        int before = instances.numInstances();
        for (int i = 0; i < instances.numAttributes(); i++)
            instances.deleteWithMissing(i);

        System.out.println((before-instances.numInstances()) + " instances with missing data removed");
    }

    public static void clusterTest(Instances instances) throws Exception {
        //used to test general clustering results. True testing occurs through ClassificationViaClustering in classificationTest method

        Clusterer clusterer;
//        clusterer = new MakeDensityBasedClusterer();
        clusterer = new SimpleKMeans();

        ClusterEvaluation evaluation = new ClusterEvaluation();
        evaluation.setClusterer(clusterer);

        Remove remove = new Remove();
        remove.setAttributeIndices(String.valueOf(instances.classIndex()+1));
        remove.setInputFormat(instances);

        Instances train = Filter.useFilter(instances, remove); //remove class to use for clustering
        clusterer.buildClusterer(train);

        evaluation.evaluateClusterer(instances); //test against class data
        System.out.println(evaluation.clusterResultsToString());
    }

    public static Instances preprocess(Instances instances) throws Exception {
        Remove remove = new Remove();
        remove.setAttributeIndices("13, 20,21,25"); //removes team name, team ID, shot ID, and minutes remaining
        remove.setInputFormat(instances);

        NumericToNominal toNominal = new NumericToNominal();
        toNominal.setAttributeIndices("15"); //make class label nominal in order to classify
        toNominal.setInputFormat(instances);

        instances = Filter.useFilter(instances, toNominal);
        instances = Filter.useFilter(instances, remove);

        return instances;
    }

    public static Instances convertDateFormat(Instances instances) throws Exception {
//        used to make date into a continuous value

        int newDateAttr = 21;
        int oldDateAttr = instances.attribute("game_date").index();
        Attribute attribute = new Attribute("date_in_ms");
        instances.insertAttributeAt(attribute, newDateAttr);
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyy-MM-dd");

        for (int i = 0; i < instances.numInstances(); i++) {
            instances.instance(i).setValue(newDateAttr, dateFormat.parse(instances.instance(i).stringValue(oldDateAttr)).getTime());
        }

//        remove old date attribute
        Remove remove = new Remove();
        remove.setAttributeIndices(String.valueOf(oldDateAttr+1));
        remove.setInputFormat(instances);

        return Filter.useFilter(instances, remove);
    }
}
