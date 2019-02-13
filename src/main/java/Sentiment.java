import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.neural.rnn.RNNCoreAnnotations;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.sentiment.SentimentCoreAnnotations;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.CoreMap;
import java.util.List;
import java.util.Properties;
import java.util.regex.Pattern;
import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

public class Sentiment {
 static StanfordCoreNLP pipeline;
 public static void init() {
  Properties props = new Properties();
  props.setProperty("annotators", "tokenize, ssplit , parse, sentiment");
  pipeline = new StanfordCoreNLP(props);
 }

 public static JSONArray findSentiment(String line) {
  String[] s = line.split(Pattern.quote("but"));
  line = line.replaceAll("but", ".");
  int jsonval = 0;
  int mainSentiment = 0;
  JSONArray arr = new JSONArray();
  if (line != null && line.length() > 0) {
   int longest = 0;
   Annotation annotation = new Annotation(line);
   pipeline.annotate(annotation);
   List < CoreMap > sentences = annotation.get(SentencesAnnotation.class);
   for (CoreMap sentence: sentences) {
    JSONObject obj = new JSONObject();
    Tree tree = sentence.get(SentimentCoreAnnotations.SentimentAnnotatedTree.class);
    int sentiment = RNNCoreAnnotations.getPredictedClass(tree);
    obj.put("Sentence", sentence);
    obj.put("Sentiment", sentiment);
    arr.add(jsonval, obj);
    jsonval = jsonval + 1;
    String partText = sentence.toString();
    if (partText.length() > longest) {
     mainSentiment = sentiment;
     longest = partText.length();
    }
   }
  }
  return arr;
 }
}
