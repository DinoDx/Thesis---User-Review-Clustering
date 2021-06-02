package ARdoc;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;
import org.ardoc.Parser;
import org.ardoc.Result;
import org.ardoc.UnknownCombinationException;


public class Filter {
    public static void main(String[] args){

        ArrayList<String> reviews = new ArrayList<>();

        try{
            File file = new File("C:\\Users\\Dino\\Desktop\\User-Review-Clustering\\reviews.txt");
            Scanner scanner = new Scanner(file);

            while(scanner.hasNextLine()){
                reviews.add(scanner.nextLine());
            }
            scanner.close();
        }catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
          }

        Parser p = Parser.getInstance();
        int i = 0;
        try{
            File file2 = new File("C:\\Users\\Dino\\Desktop\\User-Review-Clustering\\filteredData.txt");
            BufferedWriter printer = new BufferedWriter(new FileWriter(file2, true));
            for(String review : reviews){
                ArrayList<Result> res = p.extract("TA", review);
                i++;
                for(Result r : res){
                    if(r.getSentenceClass() == "PROBLEM DISCOVERY" || r.getSentenceClass() == "FEATURE REQUEST"){
                        printer.write(r.getSentence() + "\n");
                        System.out.println(i);
                    }
                }
            }
            printer.close();
        }catch(UnknownCombinationException | FileNotFoundException e){
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }   
}

