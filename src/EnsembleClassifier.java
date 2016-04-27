import weka.classifiers.Classifier;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.ClassificationViaClustering;
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.trees.J48;
import weka.clusterers.MakeDensityBasedClusterer;
import weka.clusterers.SimpleKMeans;
import weka.core.Instance;
import weka.core.Instances;

/**
 * Created by shureedkabir on 4/26/16.
 */
public class EnsembleClassifier extends Classifier{

    Classifier[] classifiers;

    public EnsembleClassifier() {
        super();
        classifiers = new Classifier[]{new DecisionTable(), new J48(), new NaiveBayes(), new ClassificationViaClustering(), new ClassificationViaClustering()};
        ((ClassificationViaClustering)classifiers[3]).setClusterer(new MakeDensityBasedClusterer());
        ((ClassificationViaClustering)classifiers[4]).setClusterer(new SimpleKMeans());
    }

    @Override
    public void buildClassifier(Instances instances) throws Exception {

        for (int i = 0; i < classifiers.length; i++) {
            classifiers[i].buildClassifier(instances);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double total = 0;

        for (int i = 0; i < classifiers.length; i++) {
            total += classifiers[i].classifyInstance(instance);
        }

        return (total >= classifiers.length/2) ? 1 : 0;
    }
}
