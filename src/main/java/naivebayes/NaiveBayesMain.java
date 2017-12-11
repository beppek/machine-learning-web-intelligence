package naivebayes;

public class NaiveBayesMain {
    public static void main(String[] args) {
        NaiveBayesWeka wnb = new NaiveBayesWeka("data/wikipedia_70.arff");
        wnb.train();
        wnb.test();
    }
}
