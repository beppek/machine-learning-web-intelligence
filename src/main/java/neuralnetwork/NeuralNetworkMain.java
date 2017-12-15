package neuralnetwork;

public class NeuralNetworkMain {
    public static void main(String[] args) {
        try {
            NeuralNetwork n = new NeuralNetwork("data/matchmaker_fixed.arff");
            n.train();
            n.test();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
