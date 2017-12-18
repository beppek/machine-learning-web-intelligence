package decisiontree;

public class DecisonTreeMain {

    public static void main(String[] args) {
        try {
            DecisionTree dt = new DecisionTree("data/FIFA_skill.arff");
            dt.train();
            dt.test();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
