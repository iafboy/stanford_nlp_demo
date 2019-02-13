import static spark.Spark.*;

public class Main {
	 /** Usage: java -cp "*" StanfordCoreNlpDemo [inputFile [outputTextFile [outputXmlFile]]] */
	  public static void main(String[] args)  {
		Sentiment.init();
		
		try{
			port(8080);
			post("/hello/", (request, response) -> {
			    return Sentiment.findSentiment(request.body());
			});
			get("/hello/:name", (request, response) -> {
				 return Sentiment.findSentiment(request.params("name"));
			});
			
		}catch(Exception e){
			e.printStackTrace();
		}
	  }
}