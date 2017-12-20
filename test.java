import java.util.*;

public class test {
  
  
  
  public static void main(String [] args) {
    
    int [][] rewards;//state - action
    double [][][] transition;//original state - action - new state
    double discount;
    int iterationLimits;
    int numberStates;
    
    rewards = new int [][] {
      {2,3},
      {0,1},
      {0,1}
    };
    
    transition = new double [][][] {
      {{0.9, 0.1, 0},{0.7, 0.3, 0}}, 
      {{0.3,0.6,0.1},{0,0,1}},
      {{0.6,0,0.4},{0,0,1}}};
    
    discount = 0.8;
    iterationLimits = 100;
    numberStates = 3;
    
    MDP test_value = new MDP(numberStates, rewards, transition, discount, iterationLimits);
    test_value.valueIteration();
    System.out.println("Policies:");
    test_value.printPolicy(); 
    System.out.println("Values:");
    test_value.printValue();
    
    System.out.println();
    MDP test = new MDP(numberStates, rewards, transition, discount, iterationLimits);
    test.policyIteration(); 
    System.out.println("Policies:");
    test.printPolicy();
    System.out.println("Values:");
    test.printValue();
    
  }  
}

//algorithms

class MDP {
  
  double [] value;//0 on the hill, 1 rolling down, 2 at the bottom
  int [] policy;//0 on the hill, 1 rolling down, 2 at the bottom
  
  int [][] rewards;//state - action
  double [][][] transition;//original state - action - new state
  double discount;
  int iterationLimits;
  int numberStates;
  
  //actions: 0 drive 1 not drive
  
  public MDP (int numberStates, 
              int [][] rewards,
              double [][][] transition,
              double discount,
              int iterationLimits) {
    
    this.numberStates = numberStates;
    value = new double [numberStates];
    policy = new int [numberStates];
    Arrays.fill(value, 0);
    this.rewards = rewards;
    this.transition = transition;
    this.discount = discount;
    this.iterationLimits = iterationLimits;
  }
  
  public void printPolicy () {
    
    for (int i = 0; i < numberStates; i ++) {
      
      System.out.print("" + policy[i] + " ");
    }
    System.out.println("");
  }
  
  public void printValue () {
    
    for (int i = 0; i < numberStates; i ++) {
      
      System.out.print("" + value[i] + " ");
    }
    System.out.println("");
  }
  
  //value iteration
  public void valueIteration () {
    
    System.out.println("Value iteration");
    for (int i = 0; i < iterationLimits; i ++) {
      
      double [] temp = new double [numberStates];
      
      for (int j = 0; j < numberStates; j ++) {
        temp[j] = calValue(j);
      }
      
      for (int j = 0; j < numberStates; j ++) {
        
        value[j] = temp[j];
      } 
    }
  }
  
  
  public double calValue(int state) { //0 current state, 1,2 other states
    
    double Drive = rewards[state][0];
    double notDrive = rewards[state][1];
    
    for (int i = 0; i < numberStates; i ++) {
      
      
      Drive += (discount * transition[state][0][i] * value[i]);
      notDrive += (discount * transition[state][1][i] * value[i]);
      
    }
    policy[state] = 1;
    if (Drive > notDrive) {
      policy[state] = 0;
    } 
    return Math.max(Drive, notDrive);
  }
  
  //policy iteration
  
  public void policyIteration () {
    
    System.out.println("Policy iteration");
    for (int i = 0; i < numberStates; i ++) {
      
      policy[i] = 1;//initial policy never drive
    }
     
    for (int i = 0; i < iterationLimits; i ++) {
      
      valueDetermine();
      for(int j = 0; j < numberStates; j++) {
        calValue(j);
      }
    }
  }

  public void valueDetermine () {

      double [] coefficients = new double [(numberStates + 1) * numberStates];
      int index = 0;

      //iterating each state
      for (int j = 0; j < numberStates; j ++) {

        //iterating all transitions
        for (int k = 0; k < numberStates; k ++) {    
          coefficients[index] = discount * transition[j][policy[j]][k];
          if (j == k) {
            coefficients[index] -= 1;
          }
          index ++;
        }
        coefficients[index] = -1 * rewards[j][policy[j]];
        index ++;
      }
      
     double [][] result = equationSolver(numberStates, coefficients);

     for (int i = 0; i < numberStates; i ++) {
      value[i] = result[i][0];
     }
  }
  
  //linear equation solver
  //credit: https://proprogramming.org/java-program-to-solve-linear-equations/
  //on the hill - rolling down - at the bottom
  public double [][] equationSolver (int n, double [] coefficients) {
    
    double [][]mat = new double[n][n];
    double [][]constants = new double[n][1];
    
    int index = 0;
    
    for(int i=0; i<n; i++)
    {
      for(int j=0; j<n; j++)
      {
        mat[i][j] = coefficients[index];
        index += 1;
      }
      constants[i][0] = coefficients[index];
      index += 1;
    }
    
    double inverted_mat[][] = invert(mat);
    
    //Multiplication of mat inverse and constants
    double result[][] = new double[n][1];
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < 1; j++)
      {
        for (int k = 0; k < n; k++)
        {
          result[i][j] = result[i][j] + inverted_mat[i][k] * constants[k][j];
        }
      }
    }
    
    return result;
  }
  
  public static double[][] invert(double a[][])
  {
    int n = a.length;
    double x[][] = new double[n][n];
    double b[][] = new double[n][n];
    int index[] = new int[n];
    for (int i=0; i<n; ++i)
      b[i][i] = 1;
    
    // Transform the matrix into an upper triangle
    gaussian(a, index);
    
    // Update the matrix b[i][j] with the ratios stored
    for (int i=0; i<n-1; ++i)
      for (int j=i+1; j<n; ++j)
      for (int k=0; k<n; ++k)
      b[index[j]][k]
      -= a[index[j]][i]*b[index[i]][k];
    
    // Perform backward substitutions
    for (int i=0; i<n; ++i)
    {
      x[n-1][i] = b[index[n-1]][i]/a[index[n-1]][n-1];
      for (int j=n-2; j>=0; --j)
      {
        x[j][i] = b[index[j]][i];
        for (int k=j+1; k<n; ++k)
        {
          x[j][i] -= a[index[j]][k]*x[k][i];
        }
        x[j][i] /= a[index[j]][j];
      }
    }
    return x;
  }
  
// Method to carry out the partial-pivoting Gaussian
// elimination.  Here index[] stores pivoting order.
  
  public static void gaussian(double a[][], int index[])
  {
    int n = index.length;
    double c[] = new double[n];
    
    // Initialize the index
    for (int i=0; i<n; ++i)
      index[i] = i;
    
    // Find the rescaling factors, one from each row
    for (int i=0; i<n; ++i)
    {
      double c1 = 0;
      for (int j=0; j<n; ++j)
      {
        double c0 = Math.abs(a[i][j]);
        if (c0 > c1) c1 = c0;
      }
      c[i] = c1;
    }
    
    // Search the pivoting element from each column
    int k = 0;
    for (int j=0; j<n-1; ++j)
    {
      double pi1 = 0;
      for (int i=j; i<n; ++i)
      {
        double pi0 = Math.abs(a[index[i]][j]);
        pi0 /= c[index[i]];
        if (pi0 > pi1)
        {
          pi1 = pi0;
          k = i;
        }
      }
      
      // Interchange rows according to the pivoting order
      int itmp = index[j];
      index[j] = index[k];
      index[k] = itmp;
      for (int i=j+1; i<n; ++i)
      {
        double pj = a[index[i]][j]/a[index[j]][j];
        
        // Record pivoting ratios below the diagonal
        a[index[i]][j] = pj;
        
        // Modify other elements accordingly
        for (int l=j+1; l<n; ++l)
          a[index[i]][l] -= pj*a[index[j]][l];
      }
    }
  }  
}


