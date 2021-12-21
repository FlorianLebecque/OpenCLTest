
using OPENCL;
using OpenTK.Compute.OpenCL;


namespace Test{

    public class Test{

        public static void Main(string[] Args){
            GPU compute = new GPU();

            float[] input  = new float[1024];
            float[] output = new float[1024];

            for(int i = 0; i < 1024; i++){
                input[i] = 1;
                output[i] = 0;
            }

            CLKernel test = compute.CreateKernel("test.cl","test");

            CLBuffer bf_input = compute.CreateBuffer<float>(MemoryFlags.ReadOnly,input);
            CLBuffer bf_ouput = compute.CreateBuffer<float>(MemoryFlags.WriteOnly,output);



            compute.SetKernelArg(test,0,bf_input);
            compute.SetKernelArg(test,1,bf_ouput);

            compute.Upload<float>(bf_input,input);

            compute.Execute(test,1,input.Count());

            compute.Download<float>(bf_ouput,output);


            foreach(float f in output){
                Console.WriteLine(f);
            }
        }

    }
    
}