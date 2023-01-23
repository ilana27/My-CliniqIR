

import com.google.common.base.Objects;

public class PubCitation2 {

private String PubType;

 private String pmid;

 private String articleTitle;

 private String abstractText;

 public PubCitation2(String pmid, String articleTitle, String PubType, String abstractText) {
 
   this.PubType = PubType;
   this.pmid = pmid;
   this.articleTitle = articleTitle;
   this.abstractText = abstractText;
 }
 
 public String getPubType() {
   return PubType;
 }

 @Override
 public String toString() {
   return String.valueOf(pmid);
 }

 @Override
 public boolean equals(Object o) {
   if (this == o)
     return true;
   if (o == null || getClass() != o.getClass())
     return false;
   PubCitation2 that = (PubCitation2) o;
   return Objects.equal(pmid, that.pmid);
 }

 @Override
 public int hashCode() {
   return Objects.hashCode(pmid);
 }

 public String getPmid() {
   return pmid;
 }

 public String getArticleTitle() {
   return articleTitle;
 }

 public String getAbstractText() {
   return abstractText;
 }

}


