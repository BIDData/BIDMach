public class getnativepath {
    public static void main(String [] args) 
    {
        String v = System.getProperty("java.library.path");
        System.out.print(v);
    }
}