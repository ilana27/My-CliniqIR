
import java.io.BufferedInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Iterator;
import java.util.List;
import java.util.zip.GZIPInputStream;

import org.jdom2.Document;
import org.jdom2.Element;
import org.jdom2.JDOMException;
import org.jdom2.input.SAXBuilder;

import com.google.common.base.Function;
import com.google.common.base.Joiner;
import com.google.common.collect.Lists;
import com.google.common.io.Files;

public class PubCitSetReader implements Iterator<PubCitation2> {

 private static final String Pubmed_CITATION_ELEMENT = "PubmedArticle";
 
 private static final String MEDLINE_CITATION_ELEMENT = "MedlineCitation";

 private static final String PMID_ELEMENT = "PMID";

 private static final String ARTICLE_ELEMENT = "Article";

 private static final String ARTICLE_TITLE_ELEMENT = "ArticleTitle";

 private static final String ABSTRACT_ELEMENT = "Abstract";

 private static final String ABSTRACT_TEXT_ELEMENT = "AbstractText";

 private static SAXBuilder builder = new SAXBuilder();

 private List<Element> citations;

 public PubCitSetReader(File file) throws IOException, JDOMException {
   String extName = Files.getFileExtension(file.getName());
   InputStream inputStream;
   if (extName.equals("xml")) {
     inputStream = new BufferedInputStream(new FileInputStream(file));
   } else if (extName.equals("gz")) {
     inputStream = new BufferedInputStream(new GZIPInputStream(new FileInputStream(file)));
   } else {
     throw new IOException("Unsupported file format.");
   }
   Document document = builder.build(inputStream);
   Element rootNode = document.getRootElement();
   citations = rootNode.getChildren(Pubmed_CITATION_ELEMENT);
   //System.out.println(citations);
 }

 public PubCitSetReader(InputStream inputStream) throws JDOMException, IOException {
   Document document = builder.build(inputStream);
   Element rootNode = document.getRootElement();
   citations = rootNode.getChildren(Pubmed_CITATION_ELEMENT); //returns PubmedArticle
   //System.out.println(citations);
   

 }

 private int idx = 0;

 @Override
 public boolean hasNext() {
   return idx < citations.size();
 }

 @Override
 public PubCitation2 next() {
   Element citationElement = citations.get(idx++);
   
  //medline citation
   
   Element medline = citationElement.getChild(MEDLINE_CITATION_ELEMENT);
 
   String pmid = (medline.getChildText(PMID_ELEMENT));
   
   
   Element articleElement = medline.getChild(ARTICLE_ELEMENT);
   // article title
   String articleTitle = articleElement.getChildText(ARTICLE_TITLE_ELEMENT);
   
   // publication type list
   Element PubList = articleElement.getChild("PublicationTypeList");
   String PubType  =  PubList.getChildText("PublicationType");
 
   
   // abstract text
   Element abstractElement = articleElement.getChild(ABSTRACT_ELEMENT);
   if (abstractElement == null) {
     return new PubCitation2(pmid, articleTitle, PubType, "");
   }
   List<String> abstractTexts = getChildrenTexts(abstractElement, ABSTRACT_TEXT_ELEMENT);
   String abstractText = Joiner.on('\n').join(abstractTexts);
   return new PubCitation2(pmid, articleTitle, PubType, abstractText);
 }

 private static List<String> getChildrenTexts(Element element, String cname) {
   List<Element> childrenElements = element.getChildren(cname);
   return Lists.transform(childrenElements, new Function<Element, String>() {

     @Override
     public String apply(Element input) {
       return input.getText();
     }
   });
 }

 @Override
 public void remove() {
   // TODO Auto-generated method stub

 }

}


