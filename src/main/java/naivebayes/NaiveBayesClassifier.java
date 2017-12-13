package naivebayes;

import core.*;

import java.util.*;

public class NaiveBayesClassifier implements Classifier {

    private int instanceCount;
    private Map<String, Double> classValues = new HashMap<>();
    private Map<String, Map<String, Map<String, Double>>> frequencyTable = new HashMap<>();

    public void train(Dataset train) {
        //Get class values
        ArrayList<String> nominalValues = train.getDistinctClassValues().getNominalValues();
        for (String nominalValue : nominalValues) {
            classValues.put(nominalValue, 0.0);
        }

        //Get attributes
        for (int i = 0; i < train.noAttributes(); i++) {
            frequencyTable.put(train.getAttributeName(i), new HashMap<>());
        }

        //Build frequency table
        instanceCount = train.noInstances();
        for (int i = 0; i < instanceCount; i++) {
            Instance instance = train.getInstance(i);
            String classValue = instance.getClassAttribute().nominalValue();
            //Increase total count for classValues
            classValues.put(classValue, classValues.get(classValue) + 1);
            ArrayList<Attribute> attributes = instance.getAttributes();
            for (int j = 0; j < instance.noAttributes(); j++) {
                String nv = attributes.get(j).nominalValue();
                String name = instance.getAttributeName(j);
                //Increase count for values based on classValues
                Map<String, Map<String, Double>> valueFrequency = frequencyTable.get(name);
                Map<String, Double> classCountForAttribute;
                //No need to store classValues
                if (classValues.containsKey(nv)) {
                    break;
                }
                //Build frequency table from ground up
                if (valueFrequency.containsKey(nv)) {
                    classCountForAttribute = valueFrequency.get(nv);
                    if (classCountForAttribute.containsKey(classValue)) {
                        classCountForAttribute.put(classValue, classCountForAttribute.get(classValue) + 1);
                    } else {
                        classCountForAttribute.put(classValue, 1.0);
                    }
                    valueFrequency.put(nv, classCountForAttribute);
                } else {
                    classCountForAttribute = new HashMap<>();
                    for (String cv : classValues.keySet()) {
                        if (cv.equals(classValue)) {
                            classCountForAttribute.put(classValue, 1.0);
                        } else {
                            classCountForAttribute.put(cv, 0.0);
                        }
                    }
                    valueFrequency.put(nv, classCountForAttribute);
                }

            }
        }

    }

    public Result classify(Instance inst) {
        Map<String, Double> probabilities = new HashMap<>();
        for (Map.Entry<String, Double> cv : classValues.entrySet()) {
            String classValue = cv.getKey();
            Double classCount = cv.getValue();

            Double pClass = classCount / instanceCount;

            probabilities.put(classValue, pClass);
            for (int i = 0; i < inst.noAttributes(); i++) {
                Attribute attribute = inst.getAttribute(i);
                String name = inst.getAttributeName(i);
                String value = attribute.nominalValue();
                if (!classValues.containsKey(value)) {
                    Double count = frequencyTable.get(name).get(value).get(classValue);
                    Double pAttribute = count / classCount;
                    probabilities.put(classValue, pAttribute * probabilities.get(classValue));
                }
            }
        }
        String result = probabilities.entrySet().stream().max(Comparator.comparing(Map.Entry::getValue)).get().getKey();
     return new Result(result);
    }

    @Override
    public String toString() {
        return frequencyTable.toString();
    }
}
