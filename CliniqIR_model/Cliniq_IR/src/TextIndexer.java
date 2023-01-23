
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FilenameFilter;


import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.index.IndexWriter;
import com.google.common.base.Stopwatch;
import com.google.common.collect.Lists;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.NumericDocValuesField;
import org.apache.lucene.document.StoredField;

import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.store.FSDirectory;
import org.jdom2.JDOMException;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.Reader;
import java.nio.file.FileAlreadyExistsException;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.ArrayList;
import java.util.List;

import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.NameValuePair;
import org.apache.http.client.HttpClient;
import org.apache.http.client.entity.UrlEncodedFormEntity;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClientBuilder;
import org.apache.http.message.BasicNameValuePair;
import org.apache.http.util.EntityUtils;

public class TextIndexer {


private final IndexWriter writer;

 public static final String PMID_FIELD = "pmid";

 public static final String ARTICLE_TITLE_FIELD = "articleTitle";

 public static final String ABSTRACT_TEXT_FIELD = "abstractText";
 
 public static final String PubType_TEXT_FIELD = "PubType";
 
 public static final String Concept_FIELD = "concept";
 
 public TextIndexer(String indexPath) throws IOException {
   File indexDir = new File(indexPath);
 
   if (!indexDir.exists()) {
     indexDir.mkdir();
   }
   Analyzer analyzer = new StandardAnalyzer();
   IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
   iwc.setOpenMode(OpenMode.CREATE_OR_APPEND);
   iwc.setRAMBufferSizeMB(2000);
   writer = new IndexWriter(FSDirectory.open(indexDir.toPath()), iwc);
 }


 public void indexDocs(File file, File filess) throws JDOMException, IOException {
 

   PubCitSetReader reader = new PubCitSetReader(filess);
   
   Scanner sc = null;
   sc= new Scanner(file);
   String input;
   StringBuffer sb = new StringBuffer();
   
           
   while (reader.hasNext() || sc.hasNextLine() )  {
   input = sc.nextLine();
   PubCitation2 citation = reader.next();
     Document doc = new Document();
     
     doc.add(new TextField(PMID_FIELD, citation.getPmid(), Field.Store.YES));
     doc.add(new TextField(ARTICLE_TITLE_FIELD, citation.getArticleTitle(), Field.Store.YES));
     String t = citation.getPubType();
     
     if (t != null) {
     doc.add(new TextField(PubType_TEXT_FIELD, citation.getPubType(), Field.Store.YES));
     }
     else {
     doc.add(new TextField(PubType_TEXT_FIELD, "Undisclosed", Field.Store.YES));
    }
     doc.add(new TextField(ABSTRACT_TEXT_FIELD, citation.getAbstractText(), Field.Store.YES));
     doc.add(new TextField(Concept_FIELD, input, Field.Store.YES));

          writer.addDocument(doc);
               
   }
   
   writer.commit();
 
   }

 public void optimize() throws IOException {
   writer.forceMerge(1);
   writer.close();
 }

 public static void main(String[] args) throws JDOMException, IOException {
 
 

 TextIndexer mci = new TextIndexer("Pubmed_Index");

 
  File dir = new File("Pubmed");
   
   List<File> files = Lists.newArrayList(dir.listFiles(new FilenameFilter() {
   

       @Override
       public boolean accept(File dir, String name) {
         return name.endsWith(".xml.gz") || name.endsWith(".xml");
       }

   }));
   
   Collections.sort(files);
  // File directoryPath = new File("ConceptDocs");
   File directoryPath = new File("/Users/tabdull1/Desktop/CliniqIR_Codes/CliniqIR_model/Pubmed_Concepts");
   List<File> files2 = Lists.newArrayList(directoryPath.listFiles(new FilenameFilter() {
   
   
     
         @Override
         public boolean accept(File dir, String name) {
             return name.endsWith(".txt") || !name.endsWith(".DS_Store");
         }
     }));
   Collections.sort(files2);
     

        List<Pair> pairs = new ArrayList<>();

        // determine result size
        int length = Math.min(files.size(), files2.size());

        // create pairs
        for (int position = 0; position < length; position++) {
            Pair pair = new Pair();
            pair.subject = files.get(position);
            pair.num = files2.get(position);
            File file = pair.subject;
            System.out.println(file.getName() + "... ");
            File filess = pair.num;
            System.out.println(filess.getName() + "... ");
           
            //////// added this
            String f1 = file.getName().split("\\.")[0];
            String f2 = filess.getName().split("\\.")[0];
           
            System.out.println(f2.equals(f1));
            if (f1.equals(f2) != true) {
            throw new java.lang.RuntimeException("this is not quite as bad");
           
            }  

     mci.indexDocs(pair.num,pair.subject);

   }
   System.out.println("Optimizing... ");

   mci.optimize();
   System.out.println("Finished");
   
   System.exit(1);
   
 }

}



