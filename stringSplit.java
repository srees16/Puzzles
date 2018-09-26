package ProjectEuler;

import java.util.*;

public class stringSplit {
	
	public static void stringSplits(List<String> dict,String s,String output) { //method to segment a String into a sequence of words in a dictionary 
        if (s.length()==0) { //if we have reached the end of the String, print the output String
            System.out.println(output);
            return;
        } else {
        	for(int i=1;i<=s.length();i++) {
                String prefix=s.substring(0,i); //consider all prefixes of current String
                if(dict.contains(prefix)) { //if the prefix is present in the dictionary,add prefix to the output String and recurse for remaining String
                	stringSplits(dict,s.substring(i),output+prefix+" ");
                }
            }
        }
    }

    public static void main(String[] args) {
        List<String> dict = Arrays.asList("snake","snakes","and","sand","ladder"); //List of Strings to represent dictionary
        String s="snakesandladder";
        System.out.println("Dictionary words are: " + dict);
        System.out.println("String is: "+s);
        stringSplits(dict,s,"");
    }
}