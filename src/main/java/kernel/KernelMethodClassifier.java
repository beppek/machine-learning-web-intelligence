package kernel;

import core.Classifier;
import core.Dataset;
import core.Instance;
import core.Result;

import java.util.ArrayList;

public class KernelMethodClassifier implements Classifier {

    private Dataset data;
    private double offset;
    private String name;

    KernelMethodClassifier(String name) {
        this.name = name;
    }

    public void train(Dataset train) {
        this.data = train;
        ArrayList<Instance> l0 = new ArrayList<>();
        ArrayList<Instance> l1 = new ArrayList<>();

        for (Instance i : data.toList()) {
            if (i.getClassAttribute().numericalValue() == 0.0) {
                l0.add(i);
            } else if (i.getClassAttribute().numericalValue() == 1.0) {
                l1.add(i);
            }
        }
        double sum0 = sumRBF(l0);
        double sum1 = sumRBF(l1);

        offset = (1.0 / Math.pow(l1.size(), 2)) * sum1 - (1.0 / Math.pow(l0.size(), 2)) * sum0;
    }

    public Result classify(Instance i) {
        double[] sum = new double[2];
        double[] count = new double[2];
        double[] point = i.getAttributeArrayNumerical();

        for (Instance inst : data.toList()) {
            double[] vals = inst.getAttributeArrayNumerical();
            int index = (int)inst.getClassAttribute().numericalValue();
            sum[index] += rbf(point, vals);
            count[index]++;
        }

        double y = (1.0 / count[0]) * sum[0] - (1.0 / count[1]) * sum[1] + offset;
        return new Result(y > 0 ? 0 : 1);
    }

    public double rbf(double[] v1, double[] v2) {
        double gamma = 20.0;
        double sq_dist = 0;
        for (int i = 0; i < v1.length; i++) {
            sq_dist += Math.pow(v1[i] - v2[i], 2);
        }
        double rb = Math.pow(Math.E, -gamma * sq_dist);
        return rb;
    }

    private double sumRBF(ArrayList<Instance> list) {
        double sum = 0;
        for (int i = 0; i < list.size(); i++) {
            double[] v1 = list.get(i).getAttributeArrayNumerical();
            for (int j = 0; j < list.size(); j++) {
                double[] v2 = list.get(j).getAttributeArrayNumerical();
                sum += rbf(v1, v2);
            }
        }
        return sum;
    }

    @Override
    public String toString() {
        return name;
    }
}
