package naivebayes;

public class NaiveBayesMain {
    public static void main(String[] args) {
        NaiveBayes wnb = new NaiveBayes("data/wikipedia_70.arff");
        wnb.train();
        wnb.test();
    }
}
