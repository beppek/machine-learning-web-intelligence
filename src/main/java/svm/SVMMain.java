package svm;

import core.Evaluator;

import java.io.File;
import java.util.Scanner;

public class SVMMain {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        File file = new File("data/matchmaker_fixed.arff");
        String absolutePath = file.getAbsolutePath();
        Evaluator evaluator = new Evaluator(new SVMClassifier("SupportVectorMachineClassifier"), absolutePath);
        evaluator.evaluateWholeSet();
        System.out.println("Press enter for 10 fold CV");
        sc.nextLine();
        evaluator.evaluateCV();
    }
}
