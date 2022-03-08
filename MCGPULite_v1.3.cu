//    --------------------------------------------------------
//    This is MCGPULite_v1.3, a modified version of the original MC-GPU v1.3.
//    Source: https://github.com/z0gSh1u/MCGPULite
//    It simplifies many procedures of the usage of MCGPU. Some behaviors are different.
//    --------------------------------------------------------

//    --> MC-GPU v1.3 was originally released in Google Code in 2012.
//        This page contains the original 2012 code, with only a bug in the function "report_voxels_dose" corrected.
//        An upgraded version of MC-GPU is being developed.

////////////////////////////////////////////////////////////////////////////////////////
//
//               ****************************
//               *** MC-GPU , version 1.3 ***
//               ****************************
//
/**
 *      \mainpage MC-GPU v1.3
 * 
 * \code   
 * 
 *               Andreu Badal, PhD (Andreu.Badal-Soler{at}fda.hhs.gov)
 * 
 *                  Division of Imaging and Applied Mathematics
 *                  Office of Science and Engineering Laboratories
 *                  Center for Devices and Radiological Health
 *                  U.S. Food and Drug Administration 
 * 
 *               Code release date: 2012/12/12
 * 
 * \endcode
 * 
 *   \b MC-GPU [1-4] is a Monte Carlo simulation code that can generate synthetic radiographic
 *   images and computed tomography (CT) scans of realistic models of the human anatomy using the
 *   computational power of commodity Graphics Processing Unit (GPU) cards.
 *   The code implements a massively multi-threaded Monte Carlo simulation algorithm
 *   for the transport of x rays in a voxelized geometry. The x ray interaction models and material
 *   properties have been adapted from \b PENELOPE \b 2006 [5].
 *
 * 
 *    \section sec_ref References
 * 
 * -# A. Badal and A. Badano, Accelerating Monte Carlo simulations of photon transport in a voxelized geometry using a massively parallel Graphics Processing Unit, Med. Phys. 36, p. 4878-4880 (2009)
 * -# A. Badal and A. Badano, Monte Carlo Simulation of X-Ray Imaging Using a Graphics Processing Unit, IEEE NSC-MIC, Conference Record , HP3–1, p. 4081-4084 (2009)
 * -# A. Badal, I. Kyprianou, D. Sharma and A. Badano, Fast cardiac CT simulation using a Graphics Processing Unit-accelerated Monte Carlo code, Proc. SPIE Medical Imaging Conference 7622, p. 762231 (2010)
 * -# A. Badal and A. Badano, Fast Simulation of Radiographic Images Using a Monte Carlo X-Ray Transport Algorithm Implemented in CUDA, Chapter 50 of GPU Computing Gems (Emerald Edition), p. 813-830, editor Wen-mei W. Hwu, publisher Morgan Kaufmann (Elsevier), Burlington MA, 2010
 * -# F. Salvat, J. M. Fernandez-Varea and J. Sempau, PENELOPE – A code system for Monte Carlo simulation of electron and photon transport, NEA-OECD, Issy-les-Moulineaux, available at www.nea.fr/html/dbprog/peneloperef.html (2006)
 * -# NVIDIA Corporation, NVIDIA CUDA(TM) Programming Guide, Technical Report available at www.nvidia.com/cuda (2011) 
 * -# A. Badal and J. Sempau, A package of Linux scripts for the parallelization of Monte Carlo simulations, Comput. Phys. Commun. 175 (6), p. 440-450 (2006) 
 * 
 * 
 *                      @file    MC-GPU_v1.3.cu
 *                      @author  Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)
 *                      @date    2012/12/12
 *                        -- MC-GPU v.1.3:  2012/12/12
 *                        -- MC-GPU v.1.2:  2011/10/25
 *                        -- MC-GPU v.1.1:  2010/06/25
 *                        -- MC-GPU v.1.0:  2009/03/17
 */ 
////////////////////////////////////////////////////////////////////////////////////////

// *** Include header file with the structures and functions declarations
#include <MCGPULite_v1.3.h>

// *** Include the computing kernel:
#include <MC-GPU_kernel_v1.3.cu>


////////////////////////////////////////////////////////////////////////////////
//!  Main program of MC-GPU: initialize the simulation enviroment, launch the GPU 
//!  kernels that perform the x ray transport and report the final results.
//!  This function reads the description of the simulation from an external file
//!  given in the command line. This input file defines the number of particles to
//!  simulate, the characteristics of the x-ray source and the detector, the number
//!  and spacing of the projections (if simulating a CT), the location of the
//!  material files containing the interaction mean free paths, and the location
//!  of the voxelized geometry file.
//!
//!                            @author  Andreu Badal
//!
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{

  // -- Start time counter:
  time_t current_time = time(NULL);             // Get current time (in seconds)  
  clock_t clock_start, clock_end, clock_start_beginning;  // (requires standard header <time.h>)
  clock_start = clock();                        // Get current clock counter
  clock_start_beginning = clock_start;
  
#ifdef USING_MPI
// -- Using MPI to access multiple GPUs to simulate the x-ray projection image:
  int myID = -88, numprocs = -99, return_reduce = -1;
  MPI_Init(&argc, &argv);                       // Init MPI and get the current thread ID 
  MPI_Comm_rank(MPI_COMM_WORLD, &myID);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  
  char MPI_processor_name[81];             
  int resultlen = -1;
  MPI_Get_processor_name(MPI_processor_name, &resultlen);
    
  char* char_time = ctime(&current_time); char_time[19] = '\0';   // The time is located betwen the characters 11 and 19.
  printf("          >> MPI run (myId=%d, numprocs=%d) on processor \"%s\" (time: %s) <<\n", myID, numprocs, MPI_processor_name, &char_time[11]);
  fflush(stdout);   // Clear the screen output buffer
  MPI_Barrier(MPI_COMM_WORLD);   // Synchronize MPI threads  
  
  MASTER_THREAD printf("              -- Time spent initializing the MPI world (MPI_Barrier): %.3f s\n", ((double)(clock()-clock_start))/CLOCKS_PER_SEC);
  
  
#else  
  int myID = 0, numprocs = 1;   // Only one CPU thread used when MPI is not activated (multiple projections will be simulated sequentially).
#endif

  MASTER_THREAD 
  { 
      printf(  "\n\033[46;30m     *****************************************************************************\033[0m\n");
      printf(    "\033[46;30m     ***     MCGPULite, version 1.3 (https://github.com/z0gSh1u/MCGPULite)     ***\033[0m\n");
      printf(    "\033[46;30m     *****************************************************************************\033[0m\n");

      printf(  "\n     *****************************************************************************\n");
      printf(    "     ***         MC-GPU, version 1.3 (http://code.google.com/p/mcgpu/)         ***\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***  A. Badal and A. Badano, \"Accelerating Monte Carlo simulations of     *** \n");
      printf(    "     ***  photon transport in a voxelized geometry using a massively parallel  *** \n");
      printf(    "     ***  Graphics Processing Unit\", Medical Physics 36, pp. 4878–4880 (2009)  ***\n");
      printf(    "     ***                                                                       ***\n");
      printf(    "     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
      printf(    "     *****************************************************************************\n\n");

      printf("****** Code execution started on: %s\n\n", ctime(&current_time));  
      fflush(stdout);
  }
    
  
#ifdef USING_CUDA
  // The "MASTER_THREAD" macro prints the messages just once when using MPI threads (it has no effect if MPI is not used):  MASTER_THREAD == "if(0==myID)"
  MASTER_THREAD printf  ("             *** CUDA SIMULATION IN THE GPU ***\n");
#else
  MASTER_THREAD printf  ("\n             *** SIMULATION IN THE CPU ***\n");
#endif

  MASTER_THREAD printf("\n    -- INITIALIZATION phase:\n");
  MASTER_THREAD fflush(stdout);   // Clear the screen output buffer for the master thread
  
  
///////////////////////////////////////////////////////////////////////////////////////////////////
  
  
  // *** Declare the arrays and structures that will contain the simulation data:

  struct voxel_struct voxel_data;          // Define the geometric constants of the voxel file
  struct detector_struct detector_data[MAX_NUM_PROJECTIONS];  // Define an x ray detector (for each projection)
  struct source_struct source_data[MAX_NUM_PROJECTIONS];      // Define the particles source (for each projection)
  struct source_energy_struct source_energy_data;    // Define the source energy spectrum
  struct linear_interp mfp_table_data;     // Constant data for the linear interpolation
  struct compton_struct  compton_table;    // Structure containing Compton sampling data (to be copied to CONSTANT memory)
  struct rayleigh_struct rayleigh_table;   // Structure containing Rayleigh sampling data (to be copied to CONSTANT memory)
  
  float2 *voxel_mat_dens = NULL;           // Poiter where voxels array will be allocated
  unsigned int voxel_mat_dens_bytes = 0;   // Size (in bytes) of the voxels array (using unsigned int to allocate up to 4.2GBytes)
  float density_max[MAX_MATERIALS];
  float density_nominal[MAX_MATERIALS];
  unsigned long long int *image = NULL;    // Poiter where image array will be allocated
  int image_bytes = -1;                    // Size of the image array
  int mfp_table_bytes = -1, mfp_Woodcock_table_bytes = -1;   // Size of the table arrays
  float2 *mfp_Woodcock_table = NULL;                // Linear interpolation data for the Woodcock mean free path [cm]
  float3 *mfp_table_a = NULL, *mfp_table_b = NULL;  // Linear interpolation data for 3 different interactions:
                                              //  (1) inverse total mean free path (divided by density, cm^2/g)
                                              //  (2) inverse Compton mean free path (divided by density, cm^2/g)
                                              //  (3) inverse Rayleigh mean free path (divided by density, cm^2/g)
  short int dose_ROI_x_min, dose_ROI_x_max, dose_ROI_y_min, dose_ROI_y_max, dose_ROI_z_min, dose_ROI_z_max;  // Coordinates of the dose region of interest (ROI)
  ulonglong2 *voxels_Edep = NULL;           // Poiter where the voxel energy deposition array will be allocated
  int voxels_Edep_bytes = 0;                      // Size of the voxel Edep array
  
  ulonglong2 materials_dose[MAX_MATERIALS];    // Array for tally_materials_dose.     !!tally_materials_dose!!
  int kk;
  for(kk=0;kk<MAX_MATERIALS;kk++) 
  {  
    materials_dose[kk].x = 0;       // Initializing data                  !!tally_materials_dose!!
    materials_dose[kk].y = 0;
    density_nominal[kk]  =-1.0f;
  }

  clock_t clock_kernel;     // Using only cpu timers after CUDA 5.0

  double time_elapsed_MC_loop = 0.0, time_total_MC_simulation = 0.0, time_total_MC_init_report = 0.0;
  

  unsigned long long int total_histories;
  int histories_per_thread, seed_input, num_threads_per_block, gpu_id, num_projections;
  int flag_material_dose = -2;
  double D_angle=-1.0, angularROI_0=0.0, angularROI_1=360.0, initial_angle=0.0, SRotAxisD=-1.0, vertical_translation_per_projection=0.0;
  char file_name_voxels[250], file_name_materials[MAX_MATERIALS][250], file_name_output[250], file_dose_output[250], file_name_espc[250];

  // *** Read the input file given in the command line and return the significant data:
  read_input(argc, argv, myID, &total_histories, &seed_input, &gpu_id, &num_threads_per_block, &histories_per_thread, detector_data, &image, &image_bytes, source_data, &source_energy_data, file_name_voxels, file_name_materials, file_name_output, file_name_espc, &num_projections, &D_angle, &angularROI_0, &angularROI_1, &initial_angle, &voxels_Edep, &voxels_Edep_bytes, file_dose_output, &dose_ROI_x_min, &dose_ROI_x_max, &dose_ROI_y_min, &dose_ROI_y_max, &dose_ROI_z_min, &dose_ROI_z_max, &SRotAxisD, &vertical_translation_per_projection, &flag_material_dose);

  // *** Read the energy spectrum and initialize its sampling with the Walker aliasing method:
  float mean_energy_spectrum = 0.0f;  
  init_energy_spectrum(file_name_espc, &source_energy_data, &mean_energy_spectrum);
  
  
  // *** Output some of the data read to make sure everything was correctly read:
  MASTER_THREAD
  {
        if (total_histories < (unsigned long long int)(100000))
          printf("                       simulation time = %lld s\n", total_histories);
        else            
          printf("              x-ray tracks to simulate = %lld\n", total_histories);
        printf("      azimuthal (phi), polar apertures = %.6f , %.6f degrees\n", ((double)source_data[0].D_phi)*RAD2DEG, 2.0*(90.0 - acos(((double)source_data[0].cos_theta_low))*RAD2DEG) );
        printf("                   focal spot position = (%f, %f, %f)\n", source_data[0].position.x, source_data[0].position.y, source_data[0].position.z);
        printf("                      source direction = (%f, %f, %f)\n", source_data[0].direction.x, source_data[0].direction.y, source_data[0].direction.z);
        printf("                  initial angle from X = %lf\n", initial_angle*RAD2DEG);
        printf("              source-detector distance = %f cm\n", detector_data[0].sdd);        
        printf("                       detector center = (%f, %f, %f)\n", (source_data[0].position.x + source_data[0].direction.x * detector_data[0].sdd),  // Center of the detector straight ahead of the focal spot.
                                                                          (source_data[0].position.y + source_data[0].direction.y * detector_data[0].sdd),
                                                                          (source_data[0].position.z + source_data[0].direction.z * detector_data[0].sdd));
        printf("                number of pixels image = %dx%d = %d\n", detector_data[0].num_pixels.x, detector_data[0].num_pixels.y, detector_data[0].total_num_pixels);
        printf("                            pixel size = %.3fx%.3f cm\n", 1.0f/detector_data[0].inv_pixel_size_X, 1.0f/detector_data[0].inv_pixel_size_Z);
        printf("                 number of projections = %d\n", num_projections);
        if (num_projections!=1)
        {
          printf("         source-rotation axis-distance = %lf cm\n", SRotAxisD);          
          printf("             angle between projections = %lf\n", D_angle*RAD2DEG);
        }
        printf("                      Input voxel file = %s\n", file_name_voxels);
        printf("                     Output image file = %s\n", file_name_output);

        printf(  "                  Energy spectrum file = %s\n", file_name_espc);      
        printf(  "             minimum, maximum energies = %.3f, %.3f keV\n", 0.001f*source_energy_data.espc[0], 0.001f*source_energy_data.espc[source_energy_data.num_bins_espc]);
        printf(  "                  mean energy spectrum = %.3f keV\n\n", 0.001f*mean_energy_spectrum);
        
        fflush(stdout);       
  }
  
 

  // *** Set the detectors and sources for the CT trajectory (if needed, ie, for more than one projection):
  if (num_projections != 1)
  {
    set_CT_trajectory(myID, num_projections, D_angle, angularROI_0, angularROI_1, SRotAxisD, source_data, detector_data, vertical_translation_per_projection);
  }
  
  fflush(stdout);
        

  // *** Read the voxel data and allocate the density map matrix. Return the maximum density:
  load_voxels(myID, file_name_voxels, density_max, &voxel_data, &voxel_mat_dens, &voxel_mat_dens_bytes, &dose_ROI_x_max, &dose_ROI_y_max, &dose_ROI_z_max);

  // *** Read the material mean free paths and set the interaction table in a "linear_interp" structure:
  load_material(myID, file_name_materials, density_max, density_nominal, &mfp_table_data, &mfp_Woodcock_table, &mfp_Woodcock_table_bytes, &mfp_table_a, &mfp_table_b, &mfp_table_bytes, &rayleigh_table, &compton_table);

  // -- Check that the input material tables and the x-ray source are consistent:
  if ( (source_energy_data.espc[0] < mfp_table_data.e0) || (source_energy_data.espc[source_energy_data.num_bins_espc] > (mfp_table_data.e0 + (mfp_table_data.num_values-1)/mfp_table_data.ide)) )
  {
    MASTER_THREAD 
    {
      printf("\n\n\n \033[31m!!!!ERROR!!\033[0m The input x-ray source energy spectrum minimum (%.3f eV) and maximum (%.3f eV) energy values\n", source_energy_data.espc[0], source_energy_data.espc[source_energy_data.num_bins_espc]);
      printf(  "           are outside the tabulated energy interval for the material properties tables (from %.3f to %.3f eV)!!\n", mfp_table_data.e0, (mfp_table_data.e0+(mfp_table_data.num_values-1)/mfp_table_data.ide));
      printf(  "           Please, modify the input energy spectra to fit the tabulated limits or create new tables.\n\n");
    }
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-1);
  }

  // -- Pre-compute the total mass of each material present in the voxel phantom (to be used in "report_materials_dose"):
  double voxel_volume = 1.0 / ( ((double)voxel_data.inv_voxel_size.x) * ((double)voxel_data.inv_voxel_size.y) * ((double)voxel_data.inv_voxel_size.z) );
  double mass_materials[MAX_MATERIALS];
  for(kk=0; kk<MAX_MATERIALS; kk++)
    mass_materials[kk] = 0.0;
  for(kk=0; kk<(voxel_data.num_voxels.x*voxel_data.num_voxels.y*voxel_data.num_voxels.z); kk++)  // For each voxel in the geometry
    mass_materials[((int)voxel_mat_dens[kk].x)-1] += ((double)voxel_mat_dens[kk].y)*voxel_volume;        // Add material mass = density*volume



  // *** Initialize the GPU using the NVIDIA CUDA libraries, if USING_CUDA parameter defined at compile time:
#ifdef USING_CUDA    
  // -- Declare the pointers to the device global memory, when using the GPU:
  float2 *voxel_mat_dens_device     = NULL,
         *mfp_Woodcock_table_device = NULL;
  float3 *mfp_table_a_device        = NULL,
         *mfp_table_b_device        = NULL;
  unsigned long long int *image_device          = NULL;
  struct rayleigh_struct *rayleigh_table_device = NULL;
  struct compton_struct  *compton_table_device  = NULL;
  ulonglong2 *voxels_Edep_device                = NULL;
  struct detector_struct *detector_data_device  = NULL;
  struct source_struct   *source_data_device    = NULL;  
  ulonglong2 *materials_dose_device = NULL;     // !!tally_materials_dose!!

  // -- Sets the CUDA enabled GPU that will be used in the simulation, and allocate and copies the simulation data in the GPU global and constant memories.
  init_CUDA_device(&gpu_id, myID, numprocs, &voxel_data, source_data, &source_energy_data, detector_data, &mfp_table_data,  /*Variables GPU constant memory*/
        voxel_mat_dens, &voxel_mat_dens_device, voxel_mat_dens_bytes,                          /*Variables GPU global memory*/
        image, &image_device, image_bytes,
        mfp_Woodcock_table, &mfp_Woodcock_table_device, mfp_Woodcock_table_bytes,
        mfp_table_a, mfp_table_b, &mfp_table_a_device, &mfp_table_b_device, mfp_table_bytes,
        &rayleigh_table, &rayleigh_table_device,
        &compton_table, &compton_table_device, &detector_data_device, &source_data_device,
        voxels_Edep, &voxels_Edep_device, voxels_Edep_bytes, &dose_ROI_x_min, &dose_ROI_x_max, &dose_ROI_y_min, &dose_ROI_y_max, &dose_ROI_z_min, &dose_ROI_z_max,
        materials_dose, &materials_dose_device, flag_material_dose, num_projections);

  // -- Constant data already moved to the GPU: clean up unnecessary RAM memory
  free(mfp_Woodcock_table);
  free(mfp_table_a);
  free(mfp_table_b);
  if (0!=myID)    // Keep the geometry data for the MPI root because the voxel densities are still needed to compute the final doses
    free(voxel_mat_dens);
    

#endif
  
  MASTER_THREAD
  {
    current_time=time(NULL);
    printf("\n    -- INITIALIZATION finished: elapsed time = %.3f s. \n\n", ((double)(clock()-clock_start))/CLOCKS_PER_SEC);
  }
  

#ifdef USING_MPI
  fflush(stdout);
  MPI_Barrier(MPI_COMM_WORLD);   // Synchronize MPI threads before starting the MC phase.
#endif

  
///////////////////////////////////////////////////////////////////////////////////////////////////
  
  
  
  MASTER_THREAD
  {
    current_time=time(NULL);
    printf("\n    -- MONTE CARLO LOOP phase. Time: %s\n", ctime(&current_time)); 
    fflush(stdout);    
  }

  
  // -- A number of histories smaller than 24 hours in sec (3600*24=86400) means that the user wants to simulate for the input number of seconds in each GPU, not a fix number of histories:
  unsigned long long int total_histories_INPUT = total_histories;    // Save the original input values to be re-used for multiple projections
  int seed_input_INPUT = seed_input, doing_speed_test = -1;  
  int simulating_by_time = 0;  // 0==false
  if (total_histories < (unsigned long long int)(95000))
    simulating_by_time = 1;    // 1=true
      


  int num_blocks_speed_test = 0;
  unsigned long long int histories_speed_test = (unsigned long long int)0, total_histories_speed_test = (unsigned long long int)0;
  float node_speed = -1.0f, total_speed = 1.0f;
  double current_angle;
  int num_p;  // == current projection number
  
   // *** CT simulation: find the current projection angle and start Monte Carlo simulation: 
   
  for (num_p=0; num_p<num_projections; num_p++)
  {
    
    // -- Check if this projection is inside the input angular region of interest (the angle can be negative, or larger than 360 in helical scans):
    current_angle = initial_angle + num_p * D_angle;       
    
    if ((current_angle < angularROI_0) || (current_angle > angularROI_1))
    {
      MASTER_THREAD printf("         << Skipping projection #%d of %d >> Angle %f degrees: outside angular region of interest.\n", num_p+1, num_projections, current_angle*RAD2DEG);
      continue;   // Cycle loop: do not simulate this projection!
    }
      
    if (num_projections!=1)
      MASTER_THREAD printf("\n\033[35m   << Simulating Projection %d of %d >> Angle: %lf degrees.\033[0m\n", num_p+1, num_projections, current_angle*RAD2DEG);          

    
    clock_start = clock();   // Start the CPU clock
    
#ifdef USING_CUDA
      
    // *** Simulate in the GPUs the input amount of time or amount of particles:
    
    // -- Estimate GPU speed to use a total simulation time or multiple GPUs:    
    
    if ( simulating_by_time==0 &&   // Simulating a fixed number of particles, not a fixed time (so performing the speed test only once)
         node_speed>0.0f &&         // Speed test already performed for a previous projection in this simulation (node_speed and total_speed variables set)
         numprocs>1)                // Using multiple GPUs (ie, multiple MPI threads)
    { 
      // -- Simulating successive projections after the first one with a fix number of particles, with multiple MPI threads: re-use the speed test results from the first projection image:
      total_histories = (unsigned long long int)(0.5 + ((double)total_histories_INPUT) * (((double)node_speed)/total_speed));  
      doing_speed_test = 0;   // No speed test for this projection.
    }
    else if ( simulating_by_time==1 || numprocs>1)
    {
      // -- Simulating with a time limit OR multiple MPI threads for the first time (num_p==0): run a speed test to calculate the speed of the current GPU and distribute the number of particles to the multiple GPUs or estimate the total number of particles required to run the input amount of time:      
      //    Note that this ELSE IF block will be skipped if we are using a single MPI thread and a fix number of particles.

      doing_speed_test = 1;   // Remember that we are performing the speed test to make sure we add the test histories to the total before the tally reports.

      if (node_speed<0.0f)    // Speed test not performed before (first projection being simulated): set num_blocks_speed_test and histories_speed_test.
      {
        num_blocks_speed_test = guestimate_GPU_performance(gpu_id);  // Guestimating a good number of blocks to estimate the speed of different generations of GPUs. Slower GPUs will simulate less particles and hopefully the fastest GPUs will not have to wait much.
        
              // !!DeBuG!! Error in code version 1.2 has been corrected here. Old code:   histories_speed_test = (unsigned long long int)(num_blocks_speed_test*num_threads_per_block)*(unsigned long long int)(histories_per_thread);
        
      }
      
      histories_speed_test = (unsigned long long int)(num_blocks_speed_test*num_threads_per_block)*(unsigned long long int)(histories_per_thread);


      // Re-load the input total number of histories and the random seed:
      total_histories = total_histories_INPUT;
      seed_input = seed_input_INPUT;                
      
      dim3  blocks_speed_test(num_blocks_speed_test, 1);
      dim3 threads_speed_test(num_threads_per_block, 1);

      
      // -- Init the current random number generator seed to avoid overlapping sequences with other MPI threads:      
      if (simulating_by_time == 1) 
        // Simulating by time: set an arbitrary huge number of particles to skip.
        update_seed_PRNG( (myID + num_p*numprocs), (unsigned long long int)(123456789012), &seed_input);     // Set the random number seed far from any other MPI thread (myID) and away from the seeds used in the previous projections (num_p*numprocs).
      else  
        // Simulating by histories
        update_seed_PRNG( (myID + num_p*numprocs), total_histories, &seed_input);   //  Using different random seeds for each projection            
      
      clock_kernel = clock();
      
      // -- Launch Monte Carlo simulation kernel for the speed test:
      track_particles<<<blocks_speed_test,threads_speed_test>>>(histories_per_thread, num_p, seed_input, image_device, voxels_Edep_device, voxel_mat_dens_device, mfp_Woodcock_table_device, mfp_table_a_device, mfp_table_b_device, rayleigh_table_device, compton_table_device, detector_data_device, source_data_device, materials_dose_device);
      
      
      #ifdef USING_MPI    
        // Find out the total number of histories simulated in the speed test by all the GPUs. Note that this MPI call will be executed in parallel with the GPU kernel because it is located before the cudaThreadSynchronize command!
      
        return_reduce = MPI_Allreduce(&histories_speed_test, &total_histories_speed_test, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);  
        if (MPI_SUCCESS != return_reduce)
          printf("\n\n \033[31m!!!!ERROR!!\033[0m Error reducing (MPI_Allreduce) the total number of histories in the speed test test??? return_reduce = %d for thread %d\n\n\n", return_reduce, myID);
        else
      #else
        total_histories_speed_test = histories_speed_test;
      #endif
            
      // [MCGPULite] warning: ‘cudaError_t cudaThreadSynchronize()’ is deprecated
      cudaDeviceSynchronize();
      // cudaThreadSynchronize();    // Force the runtime to wait until GPU kernel has completed
      getLastCudaError("\n\n !!Kernel execution failed while simulating particle tracks!! ");   // Check if the CUDA function returned any error

      float speed_test_time = float(clock()-clock_kernel)/CLOCKS_PER_SEC;

      node_speed = (float) (((double)histories_speed_test)/speed_test_time); 
      
      #ifdef USING_MPI  
        printf("                 (MPI process #%d): Estimated GPU speed = %lld hist / %.4f s = %.3f hist/s\n", myID, histories_speed_test, speed_test_time, node_speed);      
      #else
        printf("                  Estimated GPU speed = %lld hist / %.3f s = %.3f hist/s\n", histories_speed_test, speed_test_time, node_speed);        
      #endif

      
      // -- Init random number generator seed to avoid repeating the random numbers used in the speed test:
      update_seed_PRNG(1, histories_speed_test, &seed_input);
      
      if (simulating_by_time==1)
      {
        // -- Set number of histories for each GPU when simulating by time:
        if (total_histories > speed_test_time)
          total_histories = (total_histories - speed_test_time)*node_speed;    // Calculate the total number of remaining histories by "GPU speed" * "remaining time"
        else
          total_histories = 1;       // Enough particles simulated already, simulate just one more history (block) and report (kernel call would fail if total_histories < or == 0).
      }
      else
      {
        
        #ifdef USING_MPI 
          // -- Simulating a fix number of histories divided between all GPUs (execution time variable):                     
          //    Compute the fraction of the total speed that accounts for the current MPI thread:
          return_reduce = MPI_Allreduce(&node_speed, &total_speed, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);  // Sum all the times and send result to all processes
          
          if (MPI_SUCCESS != return_reduce)
            printf("\n\n \033[31m!!!!ERROR!!\033[0m Error reducing (MPI_Allreduce) the speed test results??? return_reduce = %d for thread %d\n\n\n", return_reduce, myID);
          else
            MASTER_THREAD 
            {
              printf("       -- Total speed for all GPUs (MPI_Allreduce) = %.3f hist/s; total histories simulated in the speed test (MPI_Allreduce) = %lld.\n", total_speed, total_histories_speed_test);
              printf("          The master thread will simulate %.2f%% of the x rays in the simulation.\n",  100.0f*node_speed/total_speed);
            }
        #else
          total_speed = node_speed;
        #endif

        // - Divide the remaining histories among the MPI threads (GPUs) according to their fraction of the total speed (rounding up).
        if (total_histories_speed_test < total_histories)
          total_histories = (unsigned long long int)(0.5 + ((double)(total_histories-total_histories_speed_test)) * ((double)(node_speed/total_speed)));
        else
          total_histories = numprocs;       // Enough particles simulated already, simulate just one more history (block) and report (kernel call would fail if total_histories < or == 0).
      }     
     
    }   // [Done with case of simulating projections by time or first projection by number of particles]    
    
    // else  ==>  if using only 1 GPU and a fixed number of histories the whole speed test is skipped. The random seed will be different for each projection because it is updated after calling the kernel below.
  
  
    // fflush(stdout); 
    // MPI_Barrier(MPI_COMM_WORLD);   // Synchronize MPI threads here if we want to have a better organized output text at the expense of losing some performance 


  
    // *** Perform the MC simulation itself (the speed test would be skipped for a single CPU thread using a fix number of histories):
  
    // -- Compute the number of CUDA blocks to simulate, rounding up and making sure it is below the limit of 65535 blocks.
    //    The total number of particles simulated will be increased to the nearest multiple "histories_per_thread".
    double total_threads = ceil(((double)total_histories)/((double)histories_per_thread));     // Divide the histories among GPU threads, rounding up and avoiding overflow     //  New in MC-GPU v1.4 (Mina's bug)
    int total_threads_blocks = (int)(((double)total_threads)/((double)num_threads_per_block) + 0.9990);   // Divide the GPU threads among CUDA blocks, rounding up
    if (total_threads_blocks>65535)
    {     
      #ifdef USING_MPI       
        printf("          WARNING (MPI process #%d): %d hist per thread would produce %d CUDA blocks (>65535 maximum).", myID, histories_per_thread, total_threads_blocks);
      #else
        printf("\n          WARNING: %d hist per thread would produce %d CUDA blocks, more than the maximum value of 65535.", histories_per_thread, total_threads_blocks);
      #endif
      total_threads_blocks = 65000;    // Increase the histories per thread to have exactly 65000 blocks.
      histories_per_thread = (int) ( ((double)total_histories)/((double)(total_threads_blocks*num_threads_per_block)) + 0.9990 );        
      printf(" Increasing to %d hist to run exactly %d blocks in the GPU.\n", histories_per_thread, total_threads_blocks);
    }
    else if (total_threads_blocks<1)
    {
      total_threads_blocks = 1;        // Make sure we have at least 1 block to run
    }      
    
    total_histories = ((unsigned long long int)(total_threads_blocks*num_threads_per_block))*histories_per_thread;   // Total histories will be equal or higher than the input value due to the rounding up in the division of the histories
    float total_histories_current_kernel_float = (float)total_histories;   // Keep a float approx of the num histories for the timing below
    
    #ifdef USING_MPI  
      MASTER_THREAD printf("\n");
      printf("        ==> CUDA (MPI process #%d in \"%s\"): Executing %d blocks of %d threads, with %d histories in each thread: %lld histories in total (random seed: %d).\n", myID, MPI_processor_name, total_threads_blocks, num_threads_per_block, histories_per_thread, total_histories, seed_input);
    #else
      printf("\n        ==> CUDA: Executing %d blocks of %d threads, with %d histories in each thread: %lld histories in total (random seed: %d).\n", total_threads_blocks, num_threads_per_block, histories_per_thread, total_histories, seed_input);     
    #endif
    fflush(stdout); 
    
    // -- Setup the execution parameters (Max number threads per block: 512, Max sizes each dimension of grid: 65535x65535x1)

    dim3 blocks(total_threads_blocks, 1);
    dim3 threads(num_threads_per_block, 1); 
    
    clock_kernel = clock();

    
    // *** Execute the x-ray transport kernel in the GPU ***
    track_particles<<<blocks,threads>>>(histories_per_thread, num_p, seed_input, image_device, voxels_Edep_device, voxel_mat_dens_device, mfp_Woodcock_table_device, mfp_table_a_device, mfp_table_b_device, rayleigh_table_device, compton_table_device, detector_data_device, source_data_device, materials_dose_device);
    
    
    if (1==doing_speed_test)
      total_histories += histories_speed_test;     // Speed test was done: compute the total number of histories including the particles simulated in the speed test 
      
    // -- Move the pseudo-random number generator seed ahead to skip all the random numbers generated in the current projection by this and the other
    //    "numprocs" MPI threads. Each projection will use independent seeds! (this code runs in parallel with the asynchronous GPU kernel):
    update_seed_PRNG(numprocs, total_histories, &seed_input);   // Do not repeat seed for each projection. Note that this function only updates 1 seed, the other is not computed.
              

    #ifdef USING_MPI 
      if (numprocs>1)  // Using more than 1 MPI thread:
      {
        // -- Compute the total number of histories simulated with all MPI thread, including the speed test (histories_speed_test==0 if speed test was skipped).
        //    These MPI messajes are sent concurrently with the GPU kernel computation for maximum efficiency.
        unsigned long long int current_GPU_histories = total_histories;  
        return_reduce = MPI_Reduce(&current_GPU_histories, &total_histories, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);  // Sum all the simulated particles and send to thread 0
                
        MASTER_THREAD 
        {
          if (MPI_SUCCESS != return_reduce)
            printf("\n\n \033[31m!!!!ERROR!!\033[0m Error getting the total number of particles simulated in all the GPUs (MPI_Reduce). return_reduce = %d.\n\n\n", return_reduce);
          
          if (1==simulating_by_time || 1==doing_speed_test)
          {
            printf("\n       -- Total number of histories being simulated in all the GPUs for the current projection (including speed test)= %.3lld.\n\n", total_histories);
            fflush(stdout);
          }
        }
      }
    #endif

    // [MCGPULite] warning: ‘cudaError_t cudaThreadSynchronize()’ is deprecated
    cudaDeviceSynchronize();
    // cudaThreadSynchronize();    // Force the runtime to wait until the GPU kernel is completed
    getLastCudaError("\n\n !!Kernel execution failed while simulating particle tracks!! ");  // Check if kernel execution generated any error

    float real_GPU_speed = total_histories_current_kernel_float/(float(clock()-clock_kernel)/CLOCKS_PER_SEC);  // GPU speed for all the image simulation, not just the speed test.
    
//     #ifdef USING_MPI   
//       printf("        ==> CUDA (MPI process #%d in \"%s\"): GPU kernel execution time: %.4f s (%.3f hist/s)\n", myID, MPI_processor_name, time_kernel, total_histories_current_kernel_float/time_kernel);
//     #else     
//       printf("        ==> CUDA: Kernel execution time: %.4f s\n", time_kernel);
//     #endif  
      
      
    // -- Copy the simulated image from the GPU memory to the CPU:           
    checkCudaErrors(cudaMemcpy(image, image_device, image_bytes, cudaMemcpyDeviceToHost) );  // Copy final results to host

         
///////////////////////////////////////////////////////////////////////////////////////////////////
  
    
#else
    
    // *** Executing the kernel in the CPU:
    
    //     If using more than one MPI thread, the number of particles is equally dividied among the threads.  
    //    !!DeBuG!! --> NOT USING SPEED TEST IN THE CPU!! Not possible to limit the execution by time in the CPU.
    
    int total_threads = (int)(((double)total_histories)/((double)histories_per_thread*numprocs) + 0.9990);     // Divide the histories among MPI threads, rounding up
    unsigned long long int total_histories_per_thread = ((unsigned long long int)(total_threads))*histories_per_thread;
    total_histories = total_histories_per_thread*numprocs;    // Total histories will be equal or higher than the input value due to the rounding up in the division of the histories       
    
    
    if (numprocs>1) 
    {
      #ifdef USING_MPI       
        update_seed_PRNG(myID, total_histories, &seed_input);   // Compute the initial random seed for each MPI threads, avoiding overlapping of the random sequences
      
        printf("       Executing %d history batches in the CPU, with %d histories in each batch  (thread %d of %d at \'%s\'): %lld histories (random seed=%d).\n", total_threads, histories_per_thread, myID+1, numprocs, MPI_processor_name, total_histories_per_thread, seed_input);
        MASTER_THREAD printf("       Simulating %lld histories in total for the %d MPI threads.\n\n", total_histories, numprocs);
      #endif
    }
    else
    {
      printf("       Executing %d history batches in the CPU, with %d histories in each batch: %lld histories in total.\n\n", total_threads, histories_per_thread, total_histories);
    }    
    fflush(stdout); 
    

    // -- Copy local structures to global struct variables accessible from "track_particles" (__constant__ variables in the GPU):
    source_energy_data_CONST = source_energy_data;
    voxel_data_CONST = voxel_data;
    mfp_table_data_CONST = mfp_table_data;
    dose_ROI_x_min_CONST = dose_ROI_x_min;
    dose_ROI_x_max_CONST = dose_ROI_x_max;
    dose_ROI_y_min_CONST = dose_ROI_y_min;
    dose_ROI_y_max_CONST = dose_ROI_y_max;
    dose_ROI_z_min_CONST = dose_ROI_z_min;
    dose_ROI_z_max_CONST = dose_ROI_z_max;

    
    int CPU_batch;
    for(CPU_batch=0; CPU_batch<total_threads; CPU_batch++)
    {
      // -- Simulate a particle track initializing the PRNG with the particle number 'n':
      track_particles(CPU_batch, histories_per_thread, num_p, seed_input, image, voxels_Edep, voxel_mat_dens, mfp_Woodcock_table, mfp_table_a, mfp_table_b, &rayleigh_table, &compton_table, detector_data, source_data, materials_dose);
    }

    
#endif    


    // Get current time and calculate execution time in the MC loop:
    time_elapsed_MC_loop = ((double)(clock()-clock_start))/CLOCKS_PER_SEC;       
    time_total_MC_simulation += time_elapsed_MC_loop;   // Count total time (in seconds).
        //  printf("\n    -- MONTE CARLO LOOP finished: time tallied in MAIN program: %.3f s\n\n", time_elapsed_MC_loop);
        


///////////////////////////////////////////////////////////////////////////////////////////////////
     

    // *** Move the images simulated in the GPU (or multiple CPU cores) to the host memory space:
    
#ifdef USING_MPI 
    if (numprocs>1)  // Using more than 1 MPI thread
    {
      // -- Add the images simulated in all the MPI threads:      
      MASTER_THREAD printf("\n        >>  Synchronize the MPI threads and accumulate the simulated images (MPI_Reduce).\n\n");                    
      
      // Allocate the memory for the final image in the master thread:
      unsigned long long int *image_MPI = NULL;
      MASTER_THREAD image_MPI = (unsigned long long int*) malloc(image_bytes);
      MASTER_THREAD if (image_MPI==NULL)
      {
        printf("\n\n   !!malloc ERROR!! Problem allocating the total MPI image. Out of memory??\n\n");  
        exit(-4);
      }


      // !!DeBuG!! To know how much time the threads lose waiting for other threads in the MPI_Reduce, I have to use an explicit barrier here. It may be more efficient to let the threads advance to the MPI_Reduce directly.
      clock_start = clock();      
      MPI_Barrier(MPI_COMM_WORLD);   // Synchronize MPI threads            
      
      current_time=time(NULL);      
      char_time = ctime(&current_time); char_time[19] = '\0';   // The time is located between the characters 11 and 19.
      
      
    #ifdef USING_CUDA
      if (1==doing_speed_test)    // This message will be shown only for the first projection simulated in the GPU.
        printf("        ==> CUDA (MPI process #%d in \"%s\"): GPU speed = %.4f hist/s. Time spent at MPI_Barrier waiting to add the partial images: %.6f s (time: %8s)\n", myID, MPI_processor_name, real_GPU_speed, ((double)(clock()-clock_start))/CLOCKS_PER_SEC, &char_time[11]);
    #else        
      if (-1==doing_speed_test)
      {
        printf("        ==> CUDA (MPI process #%d in \"%s\"): Time spent at MPI_Barrier waiting to add the partial images: %.6f s (time: %8s)\n", myID, MPI_processor_name, ((double)(clock()-clock_start))/CLOCKS_PER_SEC, &char_time[11]);
        doing_speed_test = 0;
      }
    #endif  
      
      
      fflush(stdout);      
      
      MASTER_THREAD clock_start = clock();
                      
      // -- Sum the pixel values from the different simulated images and send to thread 0.
      //    MPI_Reduce will act as a synchronization barrier for all the MPI threads.
      int num_pixels_image = image_bytes/((int)sizeof(unsigned long long int));   // Number of elements allocated in the "image" array.         
      return_reduce = MPI_Reduce(image, image_MPI, num_pixels_image, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD); 
      
      if (MPI_SUCCESS != return_reduce)
      {
        printf("\n\n \033[31m!!!!ERROR!!\033[0m Possible error reducing (MPI_SUM) the image results??? Returned value MPI_Reduce = %d\n\n\n", return_reduce);
      }
              
      // -- Exchange the image simulated in thread 0 for the final image from all threads, in the master thread:
      MASTER_THREAD 
      {
        free(image);
        image = image_MPI;    // point the image pointer to the new image in host memory
        image_MPI = NULL;                

        printf("\n       -- Time reducing the images simulated by all the MPI threads (MPI_Reduce) according to the master thread = %.6f s.\n", ((double)(clock()-clock_start))/CLOCKS_PER_SEC); 
      }
    }
#endif
                

    // *** Report the final results:
    char file_name_output_num_p[253];
    if (1==num_projections)
      strcpy(file_name_output_num_p, file_name_output);   // Use the input name for single projection
    else
      sprintf(file_name_output_num_p, "%s_%04d", file_name_output, num_p);   // Create the output file name with the input name + projection number (4 digits, padding with 0)

    MASTER_THREAD report_image(file_name_output_num_p, detector_data, source_data, mean_energy_spectrum, image, time_elapsed_MC_loop, total_histories, num_p, num_projections, D_angle, initial_angle, myID, numprocs);

    // *** Clear the image after reporting, unless this is the last projection to simulate:
    if (num_p<(num_projections-1))
    {
      int pixels_per_image = detector_data[0].num_pixels.x * detector_data[0].num_pixels.y;
      #ifdef USING_CUDA
        MASTER_THREAD printf("       ==> CUDA: Launching kernel to reset the device image to 0: number of blocks = %d, threads per block = 128\n", (int)(ceil(pixels_per_image/128.0f)+0.01f) );
        init_image_array_GPU<<<(int)(ceil(pixels_per_image/128.0f)+0.01f),128>>>(image_device, pixels_per_image);
        // [MCGPULite] warning: ‘cudaError_t cudaThreadSynchronize()’ is deprecated
        cudaDeviceSynchronize();
        // cudaThreadSynchronize();
        getLastCudaError("\n\n !!Kernel execution failed initializing the image array!! ");  // Check if kernel execution generated any error:
      #else        
        memset(image, 0, image_bytes);     //   Init memory space to 0.  (see http://www.lainoox.com/c-memset-examples/)
      #endif
    }
    
  }  // [Projection loop end: iterate for next CT projection angle]


///////////////////////////////////////////////////////////////////////////////////////////////////
          
     
  // *** Simulation finished! Report dose and timings and clean up.

#ifdef USING_CUDA
  // if (dose_ROI_x_max > -1)
  // {   
  //   MASTER_THREAD clock_kernel = clock();    

  //   checkCudaErrors( cudaMemcpy( voxels_Edep, voxels_Edep_device, voxels_Edep_bytes, cudaMemcpyDeviceToHost) );  // Copy final dose results to host (for every MPI threads)

  //   MASTER_THREAD printf("       ==> CUDA: Time copying dose results from device to host: %.6f s\n", float(clock()-clock_kernel)/CLOCKS_PER_SEC);
  // }
  
  if (flag_material_dose==1)
    checkCudaErrors( cudaMemcpy( materials_dose, materials_dose_device, MAX_MATERIALS*sizeof(ulonglong2), cudaMemcpyDeviceToHost) );  // Copy materials dose results to host, if tally enabled in input file.   !!tally_materials_dose!!

  // -- Clean up GPU device memory:
  clock_kernel = clock();    

  cudaFree(voxel_mat_dens_device);
  cudaFree(image_device);
  cudaFree(mfp_Woodcock_table_device);
  cudaFree(mfp_table_a_device);
  cudaFree(mfp_table_b_device);
  cudaFree(voxels_Edep_device);
  // [MCGPULite] warning: ‘cudaError_t cudaThreadExit()’ is deprecated
  checkCudaErrors( cudaDeviceReset() );
  // checkCudaErrors( cudaThreadExit() );

  // MASTER_THREAD printf("       ==> CUDA: Time freeing the device memory and ending the GPU threads: %.6f s\n", float(clock()-clock_kernel)/CLOCKS_PER_SEC);

#endif


#ifdef USING_MPI
  current_time=time(NULL);     // Get current time (in seconds)
  char_time = ctime(&current_time); char_time[19] = '\0';   // The time is located betwen the characters 11 and 19.  
  printf("        >> MPI thread %d in \"%s\" done! (local time: %s)\n", myID, MPI_processor_name, &char_time[11]);
  fflush(stdout);   // Clear the screen output buffer
#endif


  
  // *** Report the total dose for all the projections, if the tally is not disabled (must be done after MPI_Barrier to have all the MPI threads synchronized):
  MASTER_THREAD clock_start = clock(); 
  
  if (dose_ROI_x_max > -1)
  {   
    
#ifdef USING_MPI
    if (numprocs>1)
    {
      // -- Use MPI_Reduce to accumulate the dose from all projections:      
      //    Allocate memory in the root node to combine the dose results with MPI_REDUCE:
      int num_voxels_ROI = voxels_Edep_bytes/((int)sizeof(ulonglong2));   // Number of elements allocated in the "dose" array.
      ulonglong2 *voxels_Edep_total = (ulonglong2*) malloc(voxels_Edep_bytes);
      if (voxels_Edep_total==NULL)
      {
        printf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d voxels by the MPI root node for the total deposited dose (and uncertainty) array (%f Mbytes)!!\n\n", num_voxels_ROI, voxels_Edep_bytes/(1024.f*1024.f));
        exit(-2);
      }
      else
      {
        MASTER_THREAD
        {
          printf("\n        >> Array for the total deposited dose correctly allocated by the MPI root node (%f Mbytes).\n", voxels_Edep_bytes/(1024.f*1024.f));
          printf(  "           Waiting at MPI_Barrier for thread synchronization.\n");
        }
      }      
      
      
      MASTER_THREAD printf("\n        >> Calling MPI_Reduce to accumulate the dose from all projections...\n\n");    
      
      return_reduce = MPI_Reduce(voxels_Edep, voxels_Edep_total, 2*num_voxels_ROI, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);   // Sum all the doses in "voxels_Edep_total" at thread 0.
            // !!DeBuG!! I am sending a "ulonglong2" array as if it was composed of 2 "ulonglong" variables per element. There could be problems if the alignment in the structure includes some extra padding space (but it seems ok for a 64-bit computer).
      if (MPI_SUCCESS != return_reduce)
      {
        printf("\n\n \033[31m!!!!ERROR!!\033[0m Possible error reducing (MPI_SUM) the dose results??? return_reduce = %d for thread %d\n\n\n", return_reduce, myID);
      }

      // -- Exchange the dose simulated in thread 0 for the final dose from all threads  
      MASTER_THREAD
      {
        free(voxels_Edep);
        voxels_Edep = voxels_Edep_total;    // point the voxels_Edep pointer to the final voxels_Edep array in host memory
        voxels_Edep_total = NULL;           // This pointer is not needed by now
      }
    }
#endif
        
    // -- Report the total dose for all the projections:
    // [MCGPULite] Just dont report information of dose.
    // MASTER_THREAD report_voxels_dose(file_dose_output, num_projections, &voxel_data, voxel_mat_dens, voxels_Edep, time_total_MC_simulation, total_histories, dose_ROI_x_min, dose_ROI_x_max, dose_ROI_y_min, dose_ROI_y_max, dose_ROI_z_min, dose_ROI_z_max, source_data);        
  }
  
  
  // -- Report "tally_materials_dose" with data from all MPI threads, if tally enabled:
  if (flag_material_dose==1)
  {
  #ifdef USING_MPI
    ulonglong2 materials_dose_total[MAX_MATERIALS];
    return_reduce = MPI_Reduce(materials_dose, materials_dose_total, 2*MAX_MATERIALS, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);   // !!tally_materials_dose!!
  #else
    ulonglong2 *materials_dose_total = materials_dose;  // Create a dummy pointer to the materials_dose data 
  #endif
    
    // [MCGPULite] Just dont report information of dose.
    // MASTER_THREAD report_materials_dose(num_projections, total_histories, density_nominal, materials_dose_total, mass_materials);    // Report the material dose  !!tally_materials_dose!!
  }
  
  // MASTER_THREAD clock_end = clock();
  // MASTER_THREAD printf("\n\n       ==> CUDA: Time reporting the dose data: %.6f s\n", ((double)(clock_end-clock_start))/CLOCKS_PER_SEC);
  

  // *** Clean up RAM memory. If CUDA was used, the geometry and table data were already cleaned for MPI threads other than root after copying data to the GPU:
  free(voxels_Edep);
  free(image);
#ifdef USING_CUDA
  MASTER_THREAD free(voxel_mat_dens);
#else
  free(voxel_mat_dens);
  free(mfp_Woodcock_table);
  free(mfp_table_a);
  free(mfp_table_b);
#endif
   

#ifdef USING_MPI
  MPI_Finalize();   // Finalize MPI library: no more MPI calls allowed below.
#endif
  
    
  MASTER_THREAD 
  {
    printf("\n\n\n\033[34m    -- SIMULATION FINISHED!\033[0m\n");
    
    time_total_MC_init_report = ((double)(clock()-clock_start_beginning))/CLOCKS_PER_SEC;

    // -- Report total performance:
    printf("\n\n       ****** TOTAL SIMULATION PERFORMANCE ******\n\n");  
    printf(    "          >>> Execution time including initialization, transport and report: %.3f s.\n", time_total_MC_init_report);
    printf(    "          >>> Total number of simulated x rays:  %lld\n", total_histories*((unsigned long long int)num_projections));      
    if (time_total_MC_init_report>0.000001)
      printf(  "          >>> Total speed (using %d thread, including initialization time) [x-rays/s]:  %.2f\n\n", numprocs, (double)(total_histories*((unsigned long long int)num_projections))/time_total_MC_init_report);
  
    current_time=time(NULL);     // Get current time (in seconds)
    
    printf("\n****** Code execution finished on: %s\n", ctime(&current_time));
  }
  
#ifdef USING_CUDA
  cudaDeviceReset();  // Destroy the CUDA context before ending program (flush visual debugger data).
#endif

  return 0;
}





////////////////////////////////////////////////////////////////////////////////
//! Read the input file given in the command line and return the significant data.
//! Example input file:
//!
//!    1000000          [Total number of histories to simulate]
//!    geometry.vox     [Voxelized geometry file name]
//!    material.mat     [Material data file name]
//!
//!       @param[in] argc   Command line parameters
//!       @param[in] argv   Command line parameters: name of input file
//!       @param[out] total_histories  Total number of particles to simulate
//!       @param[out] seed_input   Input random number generator seed
//!       @param[out] num_threads_per_block   Number of CUDA threads for each GPU block
//!       @param[out] detector_data
//!       @param[out] image
//!       @param[out] source_data
//!       @param[out] file_name_voxels
//!       @param[out] file_name_materials
//!       @param[out] file_name_output
////////////////////////////////////////////////////////////////////////////////
void read_input(int argc, char** argv, int myID, unsigned long long int* total_histories, int* seed_input, int* gpu_id, int* num_threads_per_block, int* histories_per_thread, struct detector_struct* detector_data, unsigned long long int** image_ptr, int* image_bytes, struct source_struct* source_data, struct source_energy_struct* source_energy_data, char* file_name_voxels, char file_name_materials[MAX_MATERIALS][250] , char* file_name_output, char* file_name_espc, int* num_projections, double* D_angle, double* angularROI_0, double* angularROI_1, double* initial_angle, ulonglong2** voxels_Edep_ptr, int* voxels_Edep_bytes, char* file_dose_output, short int* dose_ROI_x_min, short int* dose_ROI_x_max, short int* dose_ROI_y_min, short int* dose_ROI_y_max, short int* dose_ROI_z_min, short int* dose_ROI_z_max, double* SRotAxisD, double* vertical_translation_per_projection, int* flag_material_dose)
{
  FILE* file_ptr = NULL;
  char new_line[250];
  char *new_line_ptr = NULL;
  double dummy_double;

  // -- Read the input file name from command line, if given (otherwise keep default value):
  if (2==argc)
  {
    file_ptr = fopen(argv[1], "r");
    if (NULL==file_ptr)
    {
      printf("\n\n   !!read_input ERROR!! Input file not found or not readable. Input file name: \'%s\'\n\n", argv[1]);      
        //  Not finalizing MPI here because we want the execution to fail if there is a problem with any MPI thread!!! MPI_Finalize();   // Finalize MPI library: no more MPI calls allowed below.
      exit(-1);
    }
  }
  else if (argc>2)
  {
    
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Too many input parameter (argc=%d)!! Provide only the input file name.\n\n", argc);    
    // Finalizing MPI because all threads will detect the same problem and fail together.
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-1);
  }
  else
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Input file name not given as an execution parameter!! Try again...\n\n");
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-1);
  }

  MASTER_THREAD printf("\n    -- Reading the input file \'%s\':\n", argv[1]);


  // -- Init. [SECTION SIMULATION CONFIG v.2009-05-12]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);    // Read full line (max. 250 characters).
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION SIMULATION CONFIG v.2009-05-12\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION SIMULATION CONFIG v.2009-05-12")==NULL);   // Skip comments and empty lines until the section begins
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%lf", &dummy_double);
    *total_histories = (unsigned long long int) (dummy_double+0.0001);  // Maximum unsigned long long value: 18446744073709551615
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", seed_input);   // Set the RANECU PRNG seed (the same seed will be used to init the 2 MLCGs in RANECU)
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", gpu_id);       // GPU NUMBER WHERE SIMULATION WILL RUN
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", num_threads_per_block);  // GPU THREADS PER CUDA BLOCK
    
#ifdef USING_CUDA
  if ((*num_threads_per_block%32)!=0)
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! The input number of GPU threads per CUDA block must be a multiple of 32 (warp size). Input value: %d !!\n\n", *num_threads_per_block);
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-2);
  }
#endif
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%d", histories_per_thread);   // HISTORIES PER GPU THREAD

    
  // -- Init. [SECTION SOURCE v.2009-05-12]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION SOURCE v.2011-07-12\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION SOURCE v.2011-07-12")==NULL);   // Skip comments and empty lines until the section begins



    
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);  // X-RAY ENERGY SPECTRUM FILE
    trim_name(new_line, file_name_espc);
    
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f %f %f", &source_data[0].position.x, &source_data[0].position.y, &source_data[0].position.z);   // SOURCE POSITION: X Y Z [cm]
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f %f %f", &source_data[0].direction.x, &source_data[0].direction.y, &source_data[0].direction.z);   // SOURCE DIRECTION COSINES: U V W
    // -- Normalize the input beam direction to 1:
    dummy_double = 1.0/sqrt((double)(source_data[0].direction.x*source_data[0].direction.x + source_data[0].direction.y*source_data[0].direction.y + source_data[0].direction.z*source_data[0].direction.z));
    source_data[0].direction.x = (float)(((double)source_data[0].direction.x)*dummy_double);
    source_data[0].direction.y = (float)(((double)source_data[0].direction.y)*dummy_double);
    source_data[0].direction.z = (float)(((double)source_data[0].direction.z)*dummy_double);
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
  

  // Read input fan beam polar (theta) and azimuthal (phi) aperture angles (deg):
  double phi_aperture, theta_aperture;
  sscanf(new_line, "%lf %lf", &phi_aperture, &theta_aperture);

  if (theta_aperture > 180.0)
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Input polar aperture must be in [0,180] deg.!\n");
    MASTER_THREAD printf("                       theta_aperture = %lf, phi_aperture = %lf\n", theta_aperture, phi_aperture);
    #ifdef USING_MPI
      MPI_Finalize();
    #endif
    exit(-2);
  }
  if (phi_aperture > 360.0)
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Input azimuthal aperture must be in [0,360] deg.!\n");
    MASTER_THREAD printf("                       theta_aperture = %lf, phi_aperture = %lf\n", theta_aperture, phi_aperture);
    #ifdef USING_MPI
      MPI_Finalize();
    #endif    
    exit(-2);
  }
  
  // Entering a negative theta_aperture or phi_aperture, the emitted fan beam will cover exactly the detector: see below.
    
  // *** RECTANGULAR BEAM INITIALIZATION: aperture initially centered at (0,1,0), ie, THETA_0=90, PHI_0=90
  //     Using the algorithm used in PENMAIN.f, from penelope 2008 (by F. Salvat).
  source_data[0].cos_theta_low = (float)( cos((90.0 - 0.5*theta_aperture)*DEG2RAD) );
  source_data[0].D_cos_theta   = (float)( -2.0*source_data[0].cos_theta_low );      // Theta aperture is symetric above and below 90 deg
  source_data[0].phi_low       = (float)( (90.0 - 0.5*phi_aperture)*DEG2RAD );
  source_data[0].D_phi         = (float)( phi_aperture*DEG2RAD );
  source_data[0].max_height_at_y1cm = (float) ( tan(0.5*theta_aperture*DEG2RAD) );

  
  // If a pencil beam is input, convert the 0 angle to a very small square beam to avoid precission errors:
  if (abs(theta_aperture) < 1.0e-7)
  {
    theta_aperture = +1.00e-7;
    source_data[0].cos_theta_low = 0.0f;  // = cos(90-0)
    source_data[0].D_cos_theta   = 0.0f;
    source_data[0].max_height_at_y1cm = 0.0f;
  }
  if (abs(phi_aperture) < 1.0e-7)
  {  
    phi_aperture = +1.00e-7;
    source_data[0].phi_low       = (float)( 90.0*DEG2RAD );
    source_data[0].D_phi         = 0.0f;    
  }  
  
  
  // -- Init. [SECTION IMAGE DETECTOR v.2009-12-02]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION IMAGE DETECTOR v.2009-12-02\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION IMAGE DETECTOR v.2009-12-02")==NULL);   // Skip comments and empty lines until the section begins
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    trim_name(new_line, file_name_output);   // OUTPUT IMAGE FILE NAME (no spaces)
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    float dummy_num_pixels_x, dummy_num_pixels_y;  // Read input pixel number as float and truncated to integer
    sscanf(new_line, "%f %f", &dummy_num_pixels_x, &dummy_num_pixels_y);   // NUMBER OF PIXELS IN THE IMAGE: Nx Nz
    detector_data[0].num_pixels.x = (int)(dummy_num_pixels_x+0.001f);
    detector_data[0].num_pixels.y = (int)(dummy_num_pixels_y+0.001f);
    detector_data[0].total_num_pixels = detector_data[0].num_pixels.x * detector_data[0].num_pixels.y;

    if (detector_data[0].total_num_pixels < 1 || detector_data[0].total_num_pixels > 99999999 )
    {
      MASTER_THREAD printf("\n\n   !!read_input ERROR!! The input number of pixels is incorrect. Input: X_pix = %d, Y_pix = %d, total_num_pix = %d!!\n\n", detector_data[0].num_pixels.x, detector_data[0].num_pixels.y, detector_data[0].total_num_pixels);
      #ifdef USING_MPI
        MPI_Finalize();
      #endif      
      exit(-2); 
    }
  
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
  sscanf(new_line, "%f %f", &detector_data[0].width_X, &detector_data[0].height_Z);   // IMAGE SIZE (width, height): Dx Dz [cm]
    detector_data[0].inv_pixel_size_X = detector_data[0].num_pixels.x / detector_data[0].width_X;
    detector_data[0].inv_pixel_size_Z = detector_data[0].num_pixels.y / detector_data[0].height_Z;

  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%f", &detector_data[0].sdd);            // SOURCE-TO-DETECTOR DISTANCE [cm] (detector set in front of the source, normal to the input direction)

    float3 detector_center;   // Center of the detector straight ahead of the focal spot.
    detector_center.x = source_data[0].position.x + source_data[0].direction.x * detector_data[0].sdd;
    detector_center.y = source_data[0].position.y + source_data[0].direction.y * detector_data[0].sdd;
    detector_center.z = source_data[0].position.z + source_data[0].direction.z * detector_data[0].sdd;
        
    if ((detector_data[0].sdd)<1.0e-6)
    {
      MASTER_THREAD printf("\n\n   !!read_input ERROR!! The source-to-detector distance must be positive. Input: sdd=%f!!\n\n", detector_data[0].sdd);
      #ifdef USING_MPI
        MPI_Finalize();
      #endif      
      exit(-2);
    }

  if ( (theta_aperture < -1.0e-7) || (phi_aperture < -1.0e-7) )   // If we enter a negative angle, the fan beam will cover exactly the detector surface.
  {
    theta_aperture= 2.0 * atan(0.5*detector_data[0].height_Z/(detector_data[0].sdd)) * RAD2DEG;   // Optimum angles
    phi_aperture  = 2.0 * atan(0.5*detector_data[0].width_X/(detector_data[0].sdd)) * RAD2DEG;

    source_data[0].cos_theta_low = (float)( cos((90.0 - 0.5*theta_aperture)*DEG2RAD) );
    source_data[0].D_cos_theta   = (float)( -2.0*source_data[0].cos_theta_low );      // Theta aperture is symetric above and below 90 deg
    source_data[0].phi_low       = (float)( (90.0 - 0.5*phi_aperture)*DEG2RAD );
    source_data[0].D_phi         = (float)( phi_aperture*DEG2RAD );
    source_data[0].max_height_at_y1cm = (float) ( tan(0.5*theta_aperture*DEG2RAD) ); 
  }
  

  // -- Init. [SECTION CT SCAN v.2011-10-25]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION CT SCAN TRAJECTORY v.2011-10-25\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION CT SCAN TRAJECTORY v.2011-10-25")==NULL);  // Skip comments and empty lines until the section begins
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
  sscanf(new_line, "%d", num_projections);     // NUMBER OF PROJECTIONS (beam must be perpendicular to Z axis, set to 1 for a single projection)
  if (0 == (*num_projections))
    *num_projections = 1;      // Zero projections has the same effect as 1 projection (ie, no CT scan rotation). Negative values are allowed and then the source rotates in opposite direction (negative angles).
  if ( (fabs(*num_projections) > 1) && (fabs(source_data[0].direction.z)>0.00001f) )
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! Sorry, but currently we can only simulate CT scans when the source direction is perpendicular to the Z axis (ie, w=0).\n\n\n");   // The reconstructed planes are always parallel to the XY plane.\n");
    #ifdef USING_MPI
      MPI_Finalize();
    #endif    
    exit(-2);
  }
  if ( fabs(*num_projections) > MAX_NUM_PROJECTIONS )
  {
    MASTER_THREAD printf("\n\n   !!read_input ERROR!! The input number of projections is too large. Increase parameter MAX_NUM_PROJECTIONS=%d in the header file and recompile.\n", MAX_NUM_PROJECTIONS);
    MASTER_THREAD printf(    "                        There is no limit in the number of projections to be simulated because the source, detector data for each projection is stored in global memory and transfered to shared memory for each projection.\n\n");
    #ifdef USING_MPI
      MPI_Finalize();
    #endif    
    exit(-2);
  }
  

  if (*num_projections!=1)
  {
    // -- Skip rest of the section if simulating a single projection:
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%lf", D_angle);   // ANGLE BETWEEN PROJECTIONS [degrees] (360/num_projections for full CT)
//     printf("       [Input] %s",new_line);
    *D_angle = (*D_angle)*DEG2RAD;      // store the angle in radians
    

    // Calculate initial source angle:
    *initial_angle = acos((double)(source_data[0].direction.x));
    if (source_data[0].direction.y<0)
      *initial_angle = -(*initial_angle);     // Correct for the fact that positive and negative angles have the same ACOS
    if (*initial_angle<0.0)
      *initial_angle = (*initial_angle) + 2.0*PI;   // Make sure the angle is not negative, between [0,360) degrees.
    *initial_angle = (*initial_angle) - PI;   // Correct the fact that the source is opposite to the detector (180 degrees difference).
    if (*initial_angle<0.0)
      *initial_angle = (*initial_angle) + 2.0*PI;   // Make sure the initial angle is not negative, between [0,360) degrees.

  
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%lf %lf", angularROI_0, angularROI_1);   // ANGLES OF INTEREST (projections outside this interval will be skipped)
    
    // if (*angularROI_0<-0.001 || *angularROI_1>360.001)
    // {
    //   MASTER_THREAD printf("\n\n   !!read_input ERROR!! The angles in the angular region of interest must be in the interval [0,360]. Input: %f, %f.\n\n\n", *angularROI_0, *angularROI_1);   // // The reconstructed planes are always parallel to the XY plane.\n");
    //   #ifdef USING_MPI
    //     MPI_Finalize();
    //   #endif
    //   exit(-2);
    // }
    
    *angularROI_0 = (*angularROI_0 - 0.00001)*DEG2RAD;   // Store the angles of interest in radians, increasing a little the interval to avoid floating point precision problems
    *angularROI_1 = (*angularROI_1 + 0.00001)*DEG2RAD;


    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%lf", SRotAxisD);   // SOURCE-TO-ROTATION AXIS DISTANCE (rotation axis parallel to Z, located between the source and the detector)
    if (*SRotAxisD<0.0 || *SRotAxisD>detector_data[0].sdd)
    {
      MASTER_THREAD printf("\n\n   !!read_input ERROR!! Invalid source-to-rotation axis distance! Input: %f (sdd=%f).\n\n\n", *SRotAxisD, detector_data[0].sdd);
      #ifdef USING_MPI
        MPI_Finalize();
      #endif      
      exit(-2);
    }
    
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    sscanf(new_line, "%lf", vertical_translation_per_projection);    // VERTICAL TRANSLATION BETWEEN PROJECTIONS (HELICAL SCAN)        
    
  }
  
  
  // [MCGPULite] Dose parameters are no more needed.
  // // -- Init. [SECTION DOSE DEPOSITION v.2012-12-12] (MC-GPU v.1.3):
  // //    Electrons are not transported and therefore we are approximating that the dose is equal to the KERMA (energy released by the photons alone).
  // //    This approximation is acceptable when there is electronic equilibrium and when the range of the secondary electrons is shorter than the voxel size.
  // //    Usually the doses will be acceptable for photon energies below 1 MeV. The dose estimates may not be accurate at the interface of low density volumes.
  // do
  // {
  //   new_line_ptr = fgets(new_line, 250, file_ptr);
  //   if (new_line_ptr==NULL)
  //   {
  //     printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION DOSE DEPOSITION v.2012-12-12\'!!\n");
  //     exit(-2);
  //   }
    
  //   if (strstr(new_line,"SECTION DOSE DEPOSITION v.2011-02-18")!=NULL)  // Detect previous version of input file
  //   {
  //     MASTER_THREAD printf("\n\n   !!read_input ERROR!! Please update the input file to the new version of MC-GPU (v1.3)!!\n\n    You simply have to change the input file text line:\n         [SECTION DOSE DEPOSITION v.2011-02-18]\n\n    for these two lines:\n         [SECTION DOSE DEPOSITION v.2012-12-12]\n         NO                              # TALLY MATERIAL DOSE? [YES/NO]\n\n");
  //     exit(-2);
  //   }
    
  // }
  // while(strstr(new_line,"SECTION DOSE DEPOSITION v.2012-12-12")==NULL);  // Skip comments and empty lines until the section begins
    

  // new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);   // TALLY MATERIAL DOSE? [YES/NO]  --> turn on/off the material dose tallied adding the Edep in each material, independently of the voxels.
  // if (0==strncmp("YE",new_line,2) || 0==strncmp("Ye",new_line,2) || 0==strncmp("ye",new_line,2))
  // {
  //   *flag_material_dose = 1;
  //   MASTER_THREAD printf("       Material dose deposition tally ENABLED.\n");
  // }
  // else if (0==strncmp("NO",new_line,2) || 0==strncmp("No",new_line,2) || 0==strncmp("no",new_line,2))
  // {
  //   *flag_material_dose = 0;  // -- NO: disabling tally
  //   MASTER_THREAD printf("       Material dose deposition tally DISABLED.\n");    
  // }
  // else
  // {
  //   MASTER_THREAD printf("\n\n   !!read_input ERROR!! Answer YES or NO in the first two line of \'SECTION DOSE DEPOSITION\' to enable or disable the material dose and 3D voxel dose tallies.\n                        Input text: %s\n\n",new_line);
  //   #ifdef USING_MPI
  //     MPI_Finalize();
  //   #endif
  //   exit(-2);
  // }        

  // new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);   // TALLY 3D VOXEL DOSE? [YES/NO] 

  // if (0==strncmp("YE",new_line,2) || 0==strncmp("Ye",new_line,2) || 0==strncmp("ye",new_line,2))
  // {
  //   // -- YES: using the tally
  //   new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); trim_name(new_line, file_dose_output);   // OUTPUT DOSE FILE NAME (no spaces)
  //   new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); sscanf(new_line, "%hd %hd", dose_ROI_x_min, dose_ROI_x_max);   // # VOXELS TO TALLY DOSE: X-index min max (first voxel has index 1)
  //   new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); sscanf(new_line, "%hd %hd", dose_ROI_y_min, dose_ROI_y_max);   // # VOXELS TO TALLY DOSE: Y-index min max
  //   new_line_ptr = fgets_trimmed(new_line, 250, file_ptr); sscanf(new_line, "%hd %hd", dose_ROI_z_min, dose_ROI_z_max);   // # VOXELS TO TALLY DOSE: Z-index min max

  //   *dose_ROI_x_min -= 1; *dose_ROI_x_max -= 1;  // -Re-scale input coordinates to have index=0 for the first voxel instead of 1.
  //   *dose_ROI_y_min -= 1; *dose_ROI_y_max -= 1;
  //   *dose_ROI_z_min -= 1; *dose_ROI_z_max -= 1;

  //   MASTER_THREAD printf("       3D voxel dose deposition tally ENABLED.\n");
  //   if ( ((*dose_ROI_x_min)>(*dose_ROI_x_max)) || ((*dose_ROI_y_min)>(*dose_ROI_y_max)) || ((*dose_ROI_z_min)>(*dose_ROI_z_max)) ||
  //         (*dose_ROI_x_min)<0 || (*dose_ROI_y_min)<0 || (*dose_ROI_z_min)<0 )
  //   {
  //     MASTER_THREAD printf("\n\n   !!read_input ERROR!! The input region-of-interst in \'SECTION DOSE DEPOSITION\' is not valid: the minimum voxel index may not be zero or larger than the maximum index.\n");
  //     MASTER_THREAD printf(  "                          Input data = X[%d,%d], Y[%d,%d], Z[%d,%d]\n\n", *dose_ROI_x_min+1, *dose_ROI_x_max+1, *dose_ROI_y_min+1, *dose_ROI_y_max+1, *dose_ROI_z_min+1, *dose_ROI_z_max+1);  // Show ROI with index=1 for the first voxel instead of 0.
  //     #ifdef USING_MPI
  //       MPI_Finalize();
  //     #endif      
  //     exit(-2);
  //   }
  //   if ( ((*dose_ROI_x_min)==(*dose_ROI_x_max)) && ((*dose_ROI_y_min)==(*dose_ROI_y_max)) && ((*dose_ROI_z_min)==(*dose_ROI_z_max)) ) 
  //   {
  //     MASTER_THREAD printf("\n\n   !!read_input!! According to the input region-of-interest in \'SECTION DOSE DEPOSITION\', only the dose in the voxel (%d,%d,%d) will be tallied.\n\n",*dose_ROI_x_min,*dose_ROI_y_min,*dose_ROI_z_min);
  //   }
    
  // }
  // else if (0==strncmp("NO",new_line,2) || 0==strncmp("No",new_line,2) || 0==strncmp("no",new_line,2))
  // {
  //   // -- NO: disabling tally
  //   MASTER_THREAD printf("       3D voxel dose deposition tally DISABLED.\n");
  //   *dose_ROI_x_min = (short int) 32500; *dose_ROI_x_max = (short int) -32500;   // Set absurd values for the ROI to make sure we never get any dose tallied
  //   *dose_ROI_y_min = (short int) 32500; *dose_ROI_y_max = (short int) -32500;   // (the maximum values for short int variables are +-32768)
  //   *dose_ROI_z_min = (short int) 32500; *dose_ROI_z_max = (short int) -32500;
  // }
  // else
  // {
  //     MASTER_THREAD printf("\n\n   !!read_input ERROR!! Answer YES or NO in the first two line of \'SECTION DOSE DEPOSITION\' to enable or disable the material dose and 3D voxel dose tallies.\n                        Input text: %s\n\n",new_line);
  //     #ifdef USING_MPI
  //       MPI_Finalize();
  //     #endif
  //     exit(-2);
  // }
  // MASTER_THREAD printf("\n");



  // -- Init. [SECTION VOXELIZED GEOMETRY FILE v.2009-11-30]:
  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION VOXELIZED GEOMETRY FILE v.2009-11-30\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION VOXELIZED GEOMETRY FILE v.2009-11-30")==NULL);   // Skip comments and empty lines until the section begins
  new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
  trim_name(new_line, file_name_voxels);   // VOXEL GEOMETRY FILE (penEasy 2008 format)

  do
  {
    new_line_ptr = fgets(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!read_input ERROR!! Input file is not readable or does not contain the string \'SECTION MATERIAL FILE LIST\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"SECTION MATERIAL")==NULL);   // Skip comments and empty lines until the section begins

  int i;
  for (i=0; i<MAX_MATERIALS; i++)
  {
    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);
    if (new_line_ptr==NULL)
      file_name_materials[i][0]='\n';   // The input file is allowed to finish without defining all the materials
    else
      trim_name(new_line, file_name_materials[i]);
  }
  // [Finish reading input file]

  /////////////////////////////////////////////////////////////////////////////

  // *** Set the rotation that will bring particles from the detector plane to +Y=(0,+1,0) through a rotation around X and around Z (counter-clock):
  double rotX, rotZ, cos_rX, cos_rZ, sin_rX, sin_rZ;
      // rotX = 1.5*PI - acos(source_data[0].direction.z);  // Rotate to +Y = (0,+1,0) --> rotX_0 = 3/2*PI == -PI/2
    rotX = acos(source_data[0].direction.z) - 0.5*PI;  // Rotate to +Y = (0,+1,0) --> rotX_0 =  -PI/2
      // rotX = 0.5*PI - acos(source_data[0].direction.z);  // Rotate to +Y = (0,+1,0) --> rotX_0 =  PI/2
  if ( (source_data[0].direction.x*source_data[0].direction.x + source_data[0].direction.y*source_data[0].direction.y) > 1.0e-8 )   // == u^2+v^2 > 0
  {
      // rotZ = 0.5*PI - acos(source_data[0].direction.x/sqrt(source_data[0].direction.x*source_data[0].direction.x + source_data[0].direction.y*source_data[0].direction.y));
    if (source_data[0].direction.y >= 0.0f)
      rotZ = 0.5*PI - acos(source_data[0].direction.x/sqrt(source_data[0].direction.x*source_data[0].direction.x + source_data[0].direction.y*source_data[0].direction.y));
    else
      rotZ = 0.5*PI - (-acos(source_data[0].direction.x/sqrt(source_data[0].direction.x*source_data[0].direction.x + source_data[0].direction.y*source_data[0].direction.y)));
  }
  else
    rotZ = 0.0;   // Vector pointing to +Z, do not rotate around Z then.
 
  // -- Set the rotation matrix RzRx (called inverse because moves from the correct position to the reference at +Y):
  cos_rX = cos(rotX);
  cos_rZ = cos(rotZ);
  sin_rX = sin(rotX);
  sin_rZ = sin(rotZ);

  // Rotation matrix RxRz:
  detector_data[0].rot_inv[0] =  cos_rZ;
  detector_data[0].rot_inv[1] = -sin_rZ;
  detector_data[0].rot_inv[2] =  0.0f;
  detector_data[0].rot_inv[3] =  cos_rX*sin_rZ;
  detector_data[0].rot_inv[4] =  cos_rX*cos_rZ;
  detector_data[0].rot_inv[5] = -sin_rX;
  detector_data[0].rot_inv[6] =  sin_rX*sin_rZ;
  detector_data[0].rot_inv[7] =  sin_rX*cos_rZ;
  detector_data[0].rot_inv[8] =  cos_rX;



  if ((source_data[0].direction.y > 0.99999f) && (*num_projections==1))
  {
    // Simulating a single projection and initial beam pointing to +Y: no rotation needed!!
    detector_data[0].rotation_flag = 0;
    detector_data[0].corner_min_rotated_to_Y.x = detector_center.x;
    detector_data[0].corner_min_rotated_to_Y.y = detector_center.y;
    detector_data[0].corner_min_rotated_to_Y.z = detector_center.z;

    MASTER_THREAD printf("       Source pointing to (0,1,0): detector not rotated, initial location in voxels found faster.\n");  // maximizing code efficiency -> the simulation will be faster than for other angles (but not much).");

  }
  else
  { // Rotation needed to set the detector perpendicular to +Y:
    detector_data[0].rotation_flag = 1;
    // -- Rotate the detector center to +Y:
    detector_data[0].corner_min_rotated_to_Y.x = detector_center.x*detector_data->rot_inv[0] + detector_center.y*detector_data[0].rot_inv[1] + detector_center.z*detector_data[0].rot_inv[2];
    detector_data[0].corner_min_rotated_to_Y.y = detector_center.x*detector_data[0].rot_inv[3] + detector_center.y*detector_data[0].rot_inv[4] + detector_center.z*detector_data[0].rot_inv[5];
    detector_data[0].corner_min_rotated_to_Y.z = detector_center.x*detector_data[0].rot_inv[6] + detector_center.y*detector_data[0].rot_inv[7] + detector_center.z*detector_data[0].rot_inv[8];

    MASTER_THREAD printf("       Rotations from the input direction to +Y [deg]: rotZ = %f , rotX = %f\n", rotZ*RAD2DEG, rotX*RAD2DEG);
    
  }
  // -- Set the lower corner (minimum) coordinates at the normalized orientation: +Y. The detector has thickness 0.
  detector_data[0].corner_min_rotated_to_Y.x = detector_data[0].corner_min_rotated_to_Y.x - 0.5*detector_data[0].width_X;
//  detector_data[0].corner_min_rotated_to_Y.y = detector_data[0].corner_min_rotated_to_Y.y;
  detector_data[0].corner_min_rotated_to_Y.z = detector_data[0].corner_min_rotated_to_Y.z - 0.5*detector_data[0].height_Z;
  
  detector_data[0].center.x = source_data[0].position.x + source_data[0].direction.x * detector_data[0].sdd;
  detector_data[0].center.y = source_data[0].position.y + source_data[0].direction.y * detector_data[0].sdd;
  detector_data[0].center.z = source_data[0].position.z + source_data[0].direction.z * detector_data[0].sdd;

  
  
  /////////////////////////////////////////////////////////////////////////////

  // *** Init the fan beam source model:

  if (1 == detector_data[0].rotation_flag)
  {
    // Initial beam NOT pointing to +Y: rotation is needed to move the sampled vector from (0,1,0) to the given direction!!
    rotX = 0.5*PI - acos(source_data[0].direction.z);         // ! Rotation about X: acos(wsrc)==theta, theta=90 for alpha=0, ie, +Y.
    rotZ = atan2(source_data[0].direction.y, source_data[0].direction.x) - 0.5*PI;   // ! Rotation about Z:  initial phi = 90 (+Y).  [ATAN2(v,u) = TAN(v/u), with the angle in the correct quadrant.
    cos_rX = cos(rotX);
    cos_rZ = cos(rotZ);
    sin_rX = sin(rotX);
    sin_rZ = sin(rotZ);
    // --Rotation around X (alpha) and then around Z (phi): Rz*Rx (oposite of detector rotation)
    source_data[0].rot_fan[0] =  cos_rZ;
    source_data[0].rot_fan[1] = -cos_rX*sin_rZ;
    source_data[0].rot_fan[2] =  sin_rX*sin_rZ;
    source_data[0].rot_fan[3] =  sin_rZ;
    source_data[0].rot_fan[4] =  cos_rX*cos_rZ;
    source_data[0].rot_fan[5] = -sin_rX*cos_rZ;
    source_data[0].rot_fan[6] =  0.0f;
    source_data[0].rot_fan[7] =  sin_rX;
    source_data[0].rot_fan[8] =  cos_rX;
    
    MASTER_THREAD printf("       Rotations from +Y to the input direction for the fan beam source model [deg]: rotZ = %f , rotX = %f\n", rotZ*RAD2DEG, rotX*RAD2DEG);
  }      


  /////////////////////////////////////////////////////////////////////////////


  // *** Allocate array for the 4 detected images (non-scattered, Compton, Rayleigh, multiple-scatter):
  int pixels_per_image = detector_data[0].num_pixels.x * detector_data[0].num_pixels.y;
  *image_bytes = 4 * pixels_per_image * sizeof(unsigned long long int);
  (*image_ptr) = (unsigned long long int*) malloc(*image_bytes);
  if (*image_ptr==NULL)
  {
    printf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d pixels for the 4 scatter images (%f Mbytes)!!\n\n", pixels_per_image, (*image_bytes)/(1024.f*1024.f));
    exit(-2);
  }
  else
  {

  }

  // *** Initialize the images to 0 in the CPU. The CUDA code will init it to 0 in the GPU global memory later, using kernel "init_image_array_GPU".
  memset(*image_ptr, 0, (*image_bytes));     // Init memory space to 0.   


  // *** Allocate dose and dose^2 array if tally active:
  int num_voxels_ROI = ((int)(*dose_ROI_x_max - *dose_ROI_x_min + 1)) * ((int)(*dose_ROI_y_max - *dose_ROI_y_min + 1)) * ((int)(*dose_ROI_z_max - *dose_ROI_z_min + 1));
  if ((*dose_ROI_x_max)>-1)
  {    
    *voxels_Edep_bytes = num_voxels_ROI * sizeof(ulonglong2);
    (*voxels_Edep_ptr) = (ulonglong2*) malloc(*voxels_Edep_bytes);
    if (*voxels_Edep_ptr==NULL)
    {
      printf("\n\n   !!malloc ERROR!! Not enough memory to allocate %d voxels for the deposited dose (and uncertainty) array (%f Mbytes)!!\n\n", num_voxels_ROI, (*voxels_Edep_bytes)/(1024.f*1024.f));
      exit(-2);
    }
    else
    {
      // [MCGPULite] No dose information.
      // MASTER_THREAD printf("       Array for the deposited dose ROI (and uncertainty) correctly allocated (%d voxels, %f Mbytes)\n", num_voxels_ROI, (*voxels_Edep_bytes)/(1024.f*1024.f));
    }
  }
  else
  {
    (*voxels_Edep_bytes) = 0;
  }
  
  // *** Initialize the voxel dose to 0 in the CPU. Not necessary for the CUDA code if dose matrix init. in the GPU global memory using a GPU kernel, but needed if using cudaMemcpy.  
  if ((*dose_ROI_x_max)>-1)
  {    
    memset(*voxels_Edep_ptr, 0, (*voxels_Edep_bytes));     // Init memory space to 0.
  }

  return;
}



////////////////////////////////////////////////////////////////////////////////
//! Extract a file name from an input text line, trimming the initial blanks,
//! trailing comment (#) and stopping at the first blank (the file name should
//! not contain blanks).
//!
//!       @param[in] input_line   Input sentence with blanks and a trailing comment
//!       @param[out] file_name   Trimmed file name
////////////////////////////////////////////////////////////////////////////////
void trim_name(char* input_line, char* file_name)
{
  int a=0, b=0;
  
  // Discard initial blanks:
  while(' '==input_line[a])
  {
    a++;
  }

  // Read file name until a blank or a comment symbol (#) is found:
  while ((' '!=input_line[a])&&('#'!=input_line[a]))
  {
    file_name[b] = input_line[a];
    b++;
    a++;
  }
  
  file_name[b] = '\0';    // Terminate output string
}

////////////////////////////////////////////////////////////////////////////////
//! Read a line of text and trim initial blancks and trailing comments (#).
//!
//!       @param[in] num   Characters to read
//!       @param[in] file_ptr   Pointer to the input file stream
//!       @param[out] trimmed_line   Trimmed line from input file, skipping empty lines and comments
////////////////////////////////////////////////////////////////////////////////
char* fgets_trimmed(char* trimmed_line, int num, FILE* file_ptr)
{
  char  new_line[250];
  char *new_line_ptr = NULL;
  int a=0, b=0;
  trimmed_line[0] = '\0';   //  Init with a mark that means no file input
  
  do
  {
    a=0; b=0;
    new_line_ptr = fgets(new_line, num, file_ptr);   // Read new line
    if (new_line_ptr != NULL)
    {
      // Discard initial blanks:
      while(' '==new_line[a])
      {
        a++;
      }
      // Read file until a comment symbol (#) or end-of-line are found:
      while (('\n'!=new_line[a])&&('#'!=new_line[a]))
      {
        trimmed_line[b] = new_line[a];
        b++;
        a++;
      }
    }
  } while(new_line_ptr!=NULL &&  '\0'==trimmed_line[0]);   // Keep reading lines until end-of-file or a line that is not empty or only comment is found
  
  trimmed_line[b] = '\0';    // Terminate output string
  return new_line_ptr;
}



////////////////////////////////////////////////////////////////////////////////
//! Read the voxel data and allocate the material and density matrix.
//! Also find and report the maximum density defined in the geometry.
//!
// -- Sample voxel geometry file:
//
//   #  (comment lines...)
//   #
//   #   Voxel order: X runs first, then Y, then Z.
//   #
//   [SECTION VOXELS HEADER v.2008-04-13]
//   411  190  113      No. OF VOXELS IN X,Y,Z
//   5.000e-02  5.000e-02  5.000e-02    VOXEL SIZE (cm) ALONG X,Y,Z
//   1                  COLUMN NUMBER WHERE MATERIAL ID IS LOCATED
//   2                  COLUMN NUMBER WHERE THE MASS DENSITY IS LOCATED
//   1                  BLANK LINES AT END OF X,Y-CYCLES (1=YES,0=NO)
//   [END OF VXH SECTION]
//   1 0.00120479
//   1 0.00120479
//   ...
//
//!       @param[in] file_name_voxels  Name of the voxelized geometry file.
//!       @param[out] density_max  Array with the maximum density for each material in the voxels.
//!       @param[out] voxel_data   Pointer to a structure containing the voxel number and size.
//!       @param[out] voxel_mat_dens_ptr   Pointer to the vector with the voxel materials and densities.
//!       @param[in] dose_ROI_x/y/z_max   Size of the dose ROI: can not be larger than the total number of voxels in the geometry.
////////////////////////////////////////////////////////////////////////////////
void load_voxels(int myID, char* file_name_voxels, float* density_max, struct voxel_struct* voxel_data, float2** voxel_mat_dens_ptr, unsigned int* voxel_mat_dens_bytes, short int* dose_ROI_x_max, short int* dose_ROI_y_max, short int* dose_ROI_z_max)
{
  char new_line[250];
  char *new_line_ptr = NULL;  
      
  MASTER_THREAD if (strstr(file_name_voxels,".zip")!=NULL)
    printf("\n\n    -- WARNING load_voxels! The input voxel file name has the extension \'.zip\'. Only \'.gz\' compression is allowed!!\n\n");     // !!zlib!!
    
  gzFile file_ptr = gzopen(file_name_voxels, "rb");  // Open the file with zlib: the file can be compressed with gzip or uncompressed.   !!zlib!!  
  
  if (file_ptr==NULL)
  {
    printf("\n\n   !! fopen ERROR load_voxels!! File %s does not exist!!\n", file_name_voxels);
    exit(-2);
  }
  MASTER_THREAD 
  {
    printf("\n    -- Reading voxel file \'%s\':\n",file_name_voxels);
    // [MCGPULite] Talk less.
    // if (strstr(file_name_voxels,".gz")==NULL)
    //   printf("         (note that MC-GPU can also read voxel and material files compressed with gzip)\n");     // !!zlib!!  
    fflush(stdout);
  }
  do
  {    
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    
    if (new_line_ptr==NULL)
    {
      MASTER_THREAD printf("\n\n   !!Reading ERROR load_voxels!! File is not readable or does not contain the string \'[SECTION VOXELS HEADER\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"[SECTION VOXELS")==NULL);   // Skip comments and empty lines until the header begins

  float3 voxel_size;
  new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!   // Read full line (max. 250 characters).
  sscanf(new_line, "%d %d %d",&voxel_data->num_voxels.x, &voxel_data->num_voxels.y, &voxel_data->num_voxels.z);
  new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
  sscanf(new_line, "%f %f %f", &voxel_size.x, &voxel_size.y, &voxel_size.z);
  do
  {
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    if (new_line_ptr==NULL)
    {
      MASTER_THREAD printf("\n\n   !!Reading ERROR load_voxels!! File is not readable or does not contain the string \'[END OF VXH SECTION]\'!!\n");
      exit(-2);
    }
  }
  while(strstr(new_line,"[END OF VXH SECTION")==NULL);   // Skip rest of the header

  // -- Store the size of the voxel bounding box (used in the source function):
  voxel_data->size_bbox.x = voxel_data->num_voxels.x * voxel_size.x;
  voxel_data->size_bbox.y = voxel_data->num_voxels.y * voxel_size.y;
  voxel_data->size_bbox.z = voxel_data->num_voxels.z * voxel_size.z;
  
  MASTER_THREAD printf("       Shape: %d x %d x %d =  %d\n", voxel_data->num_voxels.x, voxel_data->num_voxels.y, voxel_data->num_voxels.z, (voxel_data->num_voxels.x*voxel_data->num_voxels.y*voxel_data->num_voxels.z));
  MASTER_THREAD printf("       Voxel Size: %f x %f x %f cm  (voxel volume=%f cm^3)\n", voxel_size.x, voxel_size.y, voxel_size.z, voxel_size.x*voxel_size.y*voxel_size.z);
  MASTER_THREAD printf("       Bounding box: %f x %f x %f cm\n", voxel_data->size_bbox.x, voxel_data->size_bbox.y,  voxel_data->size_bbox.z);
  // printf("       The geometry must be given in two columns, with the voxel density in the second column.\n");
  // printf("       The  X,Y-cycles may, or may not, be separated by blank lines.\n");

  // -- Make sure the input number of voxels in the vox file is compatible with the input dose ROI (ROI assumes first voxel is index 0):
  if ( (*dose_ROI_x_max+1)>(voxel_data->num_voxels.x) || (*dose_ROI_y_max+1)>(voxel_data->num_voxels.y) || (*dose_ROI_z_max+1)>(voxel_data->num_voxels.z) )
  {
    MASTER_THREAD printf("\n       The input region of interest for the dose deposition is larger than the size of the voxelized geometry:\n");
    *dose_ROI_x_max = min_value(voxel_data->num_voxels.x-1, *dose_ROI_x_max);
    *dose_ROI_y_max = min_value(voxel_data->num_voxels.y-1, *dose_ROI_y_max);
    *dose_ROI_z_max = min_value(voxel_data->num_voxels.z-1, *dose_ROI_z_max);
    MASTER_THREAD printf(  "       updating the ROI max limits to fit the geometry -> dose_ROI_max=(%d, %d, %d)\n", *dose_ROI_x_max+1, *dose_ROI_y_max+1, *dose_ROI_z_max+1);         // Allowing the input of an ROI larger than the voxel volume: in this case some of the allocated memory will be wasted but the program will run ok.
  }
  
  if ( (*dose_ROI_x_max+1)==(voxel_data->num_voxels.x) && (*dose_ROI_y_max+1)==(voxel_data->num_voxels.y) && (*dose_ROI_z_max+1)==(voxel_data->num_voxels.z) )
    MASTER_THREAD printf("       The voxel dose tally ROI covers the entire voxelized phantom: the dose to every voxel will be tallied.\n");
  else
    MASTER_THREAD printf("       The voxel dose tally ROI covers only a fraction of the voxelized phantom: the dose to voxels outside the ROI will not be tallied.\n");
 
  // -- Store the inverse of the pixel sides (in cm) to speed up the particle location in voxels.
  voxel_data->inv_voxel_size.x = 1.0f/(voxel_size.x);
  voxel_data->inv_voxel_size.y = 1.0f/(voxel_size.y);
  voxel_data->inv_voxel_size.z = 1.0f/(voxel_size.z);
  
  // -- Allocate the voxel matrix and store array size:
  *voxel_mat_dens_bytes = sizeof(float2)*(voxel_data->num_voxels.x)*(voxel_data->num_voxels.y)*(voxel_data->num_voxels.z);
  *voxel_mat_dens_ptr    = (float2*) malloc(*voxel_mat_dens_bytes);
  if (*voxel_mat_dens_ptr==NULL)
  {
    printf("\n\n   !!malloc ERROR load_voxels!! Not enough memory to allocate %d voxels (%f Mbytes)!!\n\n", (voxel_data->num_voxels.x*voxel_data->num_voxels.y*voxel_data->num_voxels.z), (*voxel_mat_dens_bytes)/(1024.f*1024.f));
    exit(-2);
  }
  MASTER_THREAD printf("\n    -- Initializing the voxel material and density vector (%f Mbytes)...\n", (*voxel_mat_dens_bytes)/(1024.f*1024.f));
  MASTER_THREAD fflush(stdout);
  
  // -- Read the voxel densities:
  //   MASTER_THREAD printf("       Reading the voxel densities... ");
  int i, j, k, read_lines=0, dummy_material, read_items = -99;
  float dummy_density;
  float2 *voxels_ptr = *voxel_mat_dens_ptr;

  for (k=0; k<MAX_MATERIALS; k++)
    density_max[k] = -999.0f;   // Init array with an impossible low density value
  

  for(k=0; k<(voxel_data->num_voxels.z); k++)
  {
    for(j=0; j<(voxel_data->num_voxels.y); j++)
    {
      for(i=0; i<(voxel_data->num_voxels.x); i++)
      {
        
        do
        {
          new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
        } 
        while (('\n'==new_line[0])||('\n'==new_line[1])||('#'==new_line[0])||('#'==new_line[1]));   // Skip empty lines and comments.
        read_items = sscanf(new_line, "%d %f", &dummy_material, &dummy_density);    // Read the next 2 numbers
  
        if (read_items!=2)
          printf("\n   !!WARNING load_voxels!! Expecting to read 2 items (material and density). read_items=%d, read_lines=%d \n", read_items, read_lines);
  
        if (dummy_material>MAX_MATERIALS)
        {
          printf("\n\n   !!ERROR load_voxels!! Voxel material number too high!! #mat=%d, MAX_MATERIALS=%d, voxel number=%d\n\n", dummy_material, MAX_MATERIALS, read_lines+1);
          exit(-2);
        }
        if (dummy_material<1)
        {
          printf("\n\n   !!ERROR load_voxels!! Voxel material number can not be zero or negative!! #mat=%d, voxel number=%dd\n\n", dummy_material, read_lines+1);
          exit(-2);
        }
        
        if (dummy_density < 1.0e-9f)
        {
          printf("\n\n   !!ERROR load_voxels!! Voxel density can not be 0 or negative: #mat=%d, density=%f, voxel number=%d\n\n", dummy_material, dummy_density, read_lines+1);
          exit(-2);
        }        
        
        if (dummy_density > density_max[dummy_material-1])
          density_max[dummy_material-1] = dummy_density;  // Store maximum density for each material

        (*voxels_ptr).x = (float)(dummy_material)+0.0001f;  // Assign material value as float (the integer value will be recovered by truncation)
        (*voxels_ptr).y = dummy_density;      // Assign density value
        voxels_ptr++;                         // Move to next voxel

        read_lines++;
      }
    }
  }
  MASTER_THREAD printf("       Total number of voxels read: %d\n",read_lines);
  gzclose(file_ptr);     // Close input file    !!zlib!!
}


////////////////////////////////////////////////////////////////////////////////
//! Read the material input files and set the mean free paths and the "linear_interp" structures.
//! Find the material nominal density. Set the Woodcock trick data.
//
// -- Sample material data file (data obtained from the PENELOPE 2006 database and models):
//
//    [MATERIAL NAME]
//     Water
//    [NOMINAL DENSITY (g/cm^3)]
//     1.000
//    [NUMBER OF DATA VALUES]
//     4096
//    [MEAN FREE PATHS :: Energy (eV) || Rayleigh | Compton | Photoelectric | Pair-production | TOTAL (cm)]
//     1.00000E+03  7.27451E-01  9.43363E+01  2.45451E-04  1.00000E+35  2.45367E-04
//     5.00000E+03  1.80004E+00  8.35996E+00  2.38881E-02  1.00000E+35  2.35089E-02
//     1.00000E+04  4.34941E+00  6.26746E+00  2.02568E-01  1.00000E+35  1.87755E-01
//     ...
//     #[RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database)]
//     ...
//     #[COMPTON INTERACTIONS (relativistic impulse model with approximated one-electron analytical profiles)]
//     ...
//
//!       @param[in] file_name_materials    Array with the names of the material files.
//!       @param[in] density_max   maximum density in the geometry (needed to set Woodcock trick)
//!       @param[out] density_nominal   Array with the nominal density of the materials read
//!       @param[out] mfp_table_data   Constant values for the linear interpolation
//!       @param[out] mfp_table_a_ptr   First element for the linear interpolation.
//!       @param[out] mfp_table_b_ptr   Second element for the linear interpolation.
////////////////////////////////////////////////////////////////////////////////
void load_material(int myID, char file_name_materials[MAX_MATERIALS][250], float* density_max, float* density_nominal, struct linear_interp* mfp_table_data, float2** mfp_Woodcock_table_ptr, int* mfp_Woodcock_table_bytes, float3** mfp_table_a_ptr, float3** mfp_table_b_ptr, int* mfp_table_bytes, struct rayleigh_struct *rayleigh_table_ptr, struct compton_struct *compton_table_ptr)
{
  char new_line[250];
  char *new_line_ptr = NULL;
  int mat, i, bin, input_num_values = 0, input_rayleigh_values = 0, input_num_shells = 0;
  double delta_e=-99999.0;

  // -- Init the number of shells to 0 for all materials
  for (mat=0; mat<MAX_MATERIALS; mat++)
    compton_table_ptr->noscco[mat] = 0;
    

  // --Read the material data files:
  MASTER_THREAD printf("\n    -- Reading the material data files (MAX_MATERIALS=%d):\n", MAX_MATERIALS);
  for (mat=0; mat<MAX_MATERIALS; mat++)
  {
    if ((file_name_materials[mat][0]=='\0') || (file_name_materials[mat][0]=='\n'))  //  Empty file name
       continue;   // Re-start loop for next material

    MASTER_THREAD printf("         Mat %d: File \'%s\'\n", mat+1, file_name_materials[mat]);
//     printf("    -- Reading material file #%d: \'%s\'\n", mat, file_name_materials[mat]);

    gzFile file_ptr = gzopen(file_name_materials[mat], "rb");    // !!zlib!!  
    if (file_ptr==NULL)
    {
      printf("\n\n   !!fopen ERROR!! File %d \'%s\' does not exist!!\n", mat, file_name_materials[mat]);
      exit(-2);
    }
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   // Read full line (max. 250 characters).   //  !!zlib!!
      if (new_line_ptr==NULL)
      {
        printf("\n\n   !!Reading ERROR!! File is not readable or does not contain the string \'[NOMINAL DENSITY\'!!\n");
        exit(-2);
      }
    }
    while(strstr(new_line,"[NOMINAL DENSITY")==NULL);   // Skip rest of the header

    // Read the material nominal density:
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    sscanf(new_line, "# %f", &density_nominal[mat]);
    
    if (density_max[mat]>0)    //  Material found in the voxels
    {
      MASTER_THREAD printf("                Nominal density = %f g/cm^3; Max density in voxels = %f g/cm^3\n", density_nominal[mat], density_max[mat]);
    }
    else                       //  Material NOT found in the voxels
    {
      MASTER_THREAD printf("                This material is not used in any voxel.\n");
      
      // Do not lose time reading the data for materials not found in the voxels, except for the first one (needed to determine the size of the input data).      
      if (0 == mat)
        density_max[mat] = 0.01f*density_nominal[mat];   // Assign a small but positive density; this material will not be used anyway.
      else
        continue;     //  Move on to next material          
    }
      

    // --For the first material, set the number of energy values and allocate table arrays:
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    sscanf(new_line, "# %d", &input_num_values);
    if (0==mat)
    {
      mfp_table_data->num_values = input_num_values;

      // Allocate memory for the linear interpolation arrays:
      *mfp_Woodcock_table_bytes = sizeof(float2)*input_num_values;
      *mfp_Woodcock_table_ptr   = (float2*) malloc(*mfp_Woodcock_table_bytes);  // Allocate space for the 2 parameter table
      *mfp_table_bytes = sizeof(float3)*input_num_values*MAX_MATERIALS;
      *mfp_table_a_ptr = (float3*) malloc(*mfp_table_bytes);  // Allocate space for the 4 MFP tables
      *mfp_table_b_ptr = (float3*) malloc(*mfp_table_bytes);
      *mfp_table_bytes = sizeof(float3)*input_num_values*MAX_MATERIALS;

      if (input_num_values>MAX_ENERGYBINS_RAYLEIGH)
      {
        printf("\n\n   !!load_material ERROR!! Too many energy bins (Input bins=%d): increase parameter MAX_ENERGYBINS_RAYLEIGH=%d!!\n\n", input_num_values, MAX_ENERGYBINS_RAYLEIGH);
        exit(-2);
      }
      
      if ((NULL==*mfp_Woodcock_table_ptr)||(NULL==*mfp_table_a_ptr)||(NULL==*mfp_table_b_ptr))
      {
        printf("\n\n   !!malloc ERROR!! Not enough memory to allocate the linear interpolation data: %d bytes!!\n\n", (*mfp_Woodcock_table_bytes+2*(*mfp_table_bytes)));
        exit(-2);
      }
      else
      {
      }
      for (i=0; i<input_num_values; i++)
      {
        (*mfp_Woodcock_table_ptr)[i].x = 99999999.99f;    // Init this array with a huge MFP, the minimum values are calculated below
      }
    }
    else   // Materials after first
    {
      if (input_num_values != mfp_table_data->num_values)
      {
        printf("\n\n   !!load_material ERROR!! Incorrect number of energy values given in material \'%s\': input=%d, expected=%d\n",file_name_materials[mat], input_num_values, mfp_table_data->num_values);
        exit(-2);
      }
    }

    // -- Read the mean free paths (and Rayleigh cumulative prob):
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
    double d_energy, d_rayleigh, d_compton, d_photelectric, d_total_mfp, d_pmax, e_last=-1.0;
    
    for (i=0; i<input_num_values; i++)
    {

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line,"  %le  %le  %le  %le  %le  %le", &d_energy, &d_rayleigh, &d_compton, &d_photelectric, &d_total_mfp, &d_pmax);

      // Find and store the minimum total MFP at the current energy, for every material's maximum density:
      float temp_mfp = d_total_mfp*(density_nominal[mat])/(density_max[mat]);
      if (temp_mfp < (*mfp_Woodcock_table_ptr)[i].x)
        (*mfp_Woodcock_table_ptr)[i].x = temp_mfp;       // Store minimum total mfp [cm]

      // Store the inverse MFP data points with [num_values rows]*[MAX_MATERIALS columns]
      // Scaling the table to the nominal density so that I can re-scale in the kernel to the actual local density:
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].x = 1.0/(d_total_mfp*density_nominal[mat]);   // inverse TOTAL mfp * nominal density
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].y = 1.0/(d_compton  *density_nominal[mat]);   // inverse Compton mfp * nominal density
      (*mfp_table_a_ptr)[i*(MAX_MATERIALS)+mat].z = 1.0/(d_rayleigh *density_nominal[mat]);   // inverse Rayleigh mfp * nominal density

      rayleigh_table_ptr->pmax[i*(MAX_MATERIALS)+mat] = d_pmax;    // Store the maximum cumulative probability of atomic form factor F^2 for

      if (0==i && 0==mat)
      {
        mfp_table_data->e0  = d_energy;   // Store the first energy of the first material
      }

      if (0==i)
      {
        if (fabs(d_energy-mfp_table_data->e0)>1.0e-9)
        {
          printf("\n\n   !!load_material ERROR!! Incorrect first energy value given in material \'%s\': input=%f, expected=%f\n", file_name_materials[mat], d_energy, mfp_table_data->e0);
          exit(-2);
        }
      }
      else if (1==i)
      {
        delta_e = d_energy-e_last;
      }
      else if (i>1)
      {
        if (((fabs((d_energy-e_last)-delta_e))/delta_e)>0.001)  // Tolerate up to a 0.1% relative variation in the delta e (for each bin) to account for possible precission errors reading the energy values
        {
          printf("  !!ERROR reading material data!! The energy step between mean free path values is not constant!!\n      (maybe not enough decimals given for the energy values)\n      #value = %d, First delta: %f , New delta: %f, Energy: %f ; Rel.Dif=%f\n", i, delta_e, (d_energy-e_last), d_energy,((fabs((d_energy-e_last)-delta_e))/delta_e));
          exit(-2);
        }
      }
      e_last = d_energy;
    }
    
    // -- Store the inverse of delta energy:
    mfp_table_data->ide = 1.0f/delta_e;

    // -- Store MFP data slope 'b' (.y for Woodcock):
    for (i=0; i<(input_num_values-1); i++)
    {
      bin = i*MAX_MATERIALS+mat;                   // Set current bin, skipping MAX_MATERIALS columns
      (*mfp_table_b_ptr)[bin].x = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].x - (*mfp_table_a_ptr)[bin].x) / delta_e;
      (*mfp_table_b_ptr)[bin].y = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].y - (*mfp_table_a_ptr)[bin].y) / delta_e;
      (*mfp_table_b_ptr)[bin].z = ((*mfp_table_a_ptr)[bin+MAX_MATERIALS].z - (*mfp_table_a_ptr)[bin].z) / delta_e;
    }
    // After maximum energy (last bin), assume constant slope:
    (*mfp_table_b_ptr)[(input_num_values-1)*MAX_MATERIALS+mat] = (*mfp_table_b_ptr)[(input_num_values-2)*MAX_MATERIALS+mat];

    // -- Rescale the 'a' parameter (.x for Woodcock) as if the bin started at energy = 0: we will not have to rescale to the bin minimum energy every time
    for (i=0; i<input_num_values; i++)
    {
      d_energy = mfp_table_data->e0 + i*delta_e;   // Set current bin lowest energy value
      bin = i*MAX_MATERIALS+mat;                   // Set current bin, skipping MAX_MATERIALS columns
      (*mfp_table_a_ptr)[bin].x = (*mfp_table_a_ptr)[bin].x - d_energy*(*mfp_table_b_ptr)[bin].x;
      (*mfp_table_a_ptr)[bin].y = (*mfp_table_a_ptr)[bin].y - d_energy*(*mfp_table_b_ptr)[bin].y;
      (*mfp_table_a_ptr)[bin].z = (*mfp_table_a_ptr)[bin].z - d_energy*(*mfp_table_b_ptr)[bin].z;
    }

    // -- Reading data for RAYLEIGH INTERACTIONS (RITA sampling  of atomic form factor from EPDL database):
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      if (gzeof(file_ptr)!=0)                           //  !!zlib!!
      {
        printf("\n\n   !!End-of-file ERROR!! Rayleigh data not found: \"#[DATA VALUES...\" in file \'%s\'. Last line read: %s\n\n", file_name_materials[mat], new_line);
        exit(-2);
      }
    }
    while(strstr(new_line,"[DATA VALUES")==NULL);   // Skip all lines until this text is found
      
    new_line_ptr = gzgets(file_ptr, new_line, 250);   // Read the number of data points in Rayleigh     //  !!zlib!! 
    sscanf(new_line, "# %d", &input_rayleigh_values);
        
    if (input_rayleigh_values != NP_RAYLEIGH)
    {
      printf("\n\n   \033[31m!!!!ERROR!!\033[0m The number of values for Rayleigh sampling is different than the allocated space: input=%d, NP_RAYLEIGH=%d. File=\'%s\'\n", input_rayleigh_values, NP_RAYLEIGH, file_name_materials[mat]);
      exit(-2);
    }
    new_line_ptr = gzgets(file_ptr, new_line, 250);    // Comment line:  #[SAMPLING DATA FROM COMMON/CGRA/: X, P, A, B, ITL, ITU]     //  !!zlib!!
    for (i=0; i<input_rayleigh_values; i++)
    {
      int itlco_tmp, ituco_tmp;
      bin = NP_RAYLEIGH*mat + i;

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line,"  %e  %e  %e  %e  %d  %d", &(rayleigh_table_ptr->xco[bin]), &(rayleigh_table_ptr->pco[bin]),
                                                  &(rayleigh_table_ptr->aco[bin]), &(rayleigh_table_ptr->bco[bin]),
                                                  &itlco_tmp, &ituco_tmp);

      rayleigh_table_ptr->itlco[bin] = (unsigned char) itlco_tmp;
      rayleigh_table_ptr->ituco[bin] = (unsigned char) ituco_tmp;
                                                  
    }
    //  printf("    -- Rayleigh sampling data read. Input values = %d\n",input_rayleigh_values);

    // -- Reading COMPTON INTERACTIONS data (relativistic impulse model with approximated one-electron analytical profiles):
    do
    {
      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      if (gzeof(file_ptr)!=0)                           //  !!zlib!!
      {
        printf("\n\n   !!End-of-file ERROR!! Compton data not found: \"[NUMBER OF SHELLS]\" in file \'%s\'. Last line read: %s\n\n", file_name_materials[mat], new_line);
        exit(-2);
      }
    }
    while(strstr(new_line,"[NUMBER OF SHELLS")==NULL);   // Skip all lines until this text is found
    new_line_ptr = gzgets(file_ptr, new_line, 250);
    sscanf(new_line, "# %d", &input_num_shells);      // Read the NUMBER OF SHELLS
    if (input_num_shells>MAX_SHELLS)
    {
      printf("\n\n   \033[31m!!!!ERROR!!\033[0m Too many shells for Compton interactions in file \'%s\': input=%d, MAX_SHELLS=%d\n", file_name_materials[mat], input_num_shells, MAX_SHELLS);
      exit(-2);
    }
    compton_table_ptr->noscco[mat] = input_num_shells;   // Store number of shells for this material in structure
    new_line_ptr = gzgets(file_ptr, new_line, 250);      // Comment line:  #[SHELL INFORMATION FROM COMMON/CGCO/: FCO, UICO, FJ0, KZCO, KSCO]
    int kzco_dummy, ksco_dummy;
    for (i=0; i<input_num_shells; i++)
    {

      bin = mat + i*MAX_MATERIALS;

      new_line_ptr = gzgets(file_ptr, new_line, 250);   //  !!zlib!!
      sscanf(new_line," %e  %e  %e  %d  %d", &(compton_table_ptr->fco[bin]), &(compton_table_ptr->uico[bin]),
                                              &(compton_table_ptr->fj0[bin]), &kzco_dummy, &ksco_dummy);
    }
  
    gzclose(file_ptr);    // Material data read. Close the current material input file.           //  !!zlib!!
    
  }  // ["for" loop: continue with next material]


  // -- Store Woodcock MFP slope in component '.y':
  for (i=0; i<(mfp_table_data->num_values-1); i++)
    (*mfp_Woodcock_table_ptr)[i].y = ((*mfp_Woodcock_table_ptr)[i+1].x - (*mfp_Woodcock_table_ptr)[i].x)/delta_e;

  // -- Rescale the first parameter in component .x for Woodcock
  for (i=0; i<mfp_table_data->num_values; i++)
  {
    (*mfp_Woodcock_table_ptr)[i].x = (*mfp_Woodcock_table_ptr)[i].x - (mfp_table_data->e0 + i*delta_e)*(*mfp_Woodcock_table_ptr)[i].y;
  }
  
}
////////////////////////////////////////////////////////////////////////////////



#ifdef USING_CUDA
////////////////////////////////////////////////////////////////////////////////
//!  Select and initialize the CUDA-enabled GPU that will be used in the simulation.
//!  Allocates and copies the simulation data in the GPU global and constant memories.
//!
////////////////////////////////////////////////////////////////////////////////
void init_CUDA_device( int* gpu_id, int myID, int numprocs,
      /*Variables to GPU constant memory:*/ struct voxel_struct* voxel_data, struct source_struct* source_data, struct source_energy_struct* source_energy_data, struct detector_struct* detector_data, struct linear_interp* mfp_table_data,
      /*Variables to GPU global memory:*/ float2* voxel_mat_dens, float2** voxel_mat_dens_device, unsigned int voxel_mat_dens_bytes,
        unsigned long long int* image, unsigned long long int** image_device, int image_bytes,
        float2* mfp_Woodcock_table, float2** mfp_Woodcock_table_device, int mfp_Woodcock_table_bytes,
        float3* mfp_table_a, float3* mfp_table_b, float3** mfp_table_a_device, float3** mfp_table_b_device, int mfp_table_bytes,
        struct rayleigh_struct* rayleigh_table, struct rayleigh_struct** rayleigh_table_device,
        struct compton_struct* compton_table, struct compton_struct** compton_table_device, 
        struct detector_struct** detector_data_device, struct source_struct** source_data_device,
        ulonglong2* voxels_Edep, ulonglong2** voxels_Edep_device, int voxels_Edep_bytes, short int* dose_ROI_x_min, short int* dose_ROI_x_max, short int* dose_ROI_y_min, short int* dose_ROI_y_max, short int* dose_ROI_z_min, short int* dose_ROI_z_max,
        ulonglong2* materials_dose, ulonglong2** materials_dose_device, int flag_material_dose, int num_projections)
{    
  cudaDeviceProp deviceProp;
  int deviceCount;  
  checkCudaErrors(cudaGetDeviceCount(&deviceCount));
  if (0==deviceCount)
  {
    printf("\n  \033[31m!!!!ERROR!!\033[0m No CUDA enabled GPU detected by thread #%d!!\n\n", myID);
    exit(-1);
  }  
  
  
#ifdef USING_MPI      
  if (numprocs>1)
  {      
    // *** Select the appropriate GPUs in the different workstations in the MPI hostfile:
    //     The idea is that each threads will wait for the previous thread to send a messages with its processor name and GPU id, 
    //     then it will assign the current GPU, and finally it will notify the following thread:    
    const int NODE_NAME_LENGTH = 31;
    char processor_name[NODE_NAME_LENGTH+1], previous_processor_name[NODE_NAME_LENGTH+1];
    int resultlen = -1;
    
    MPI_Get_processor_name(processor_name, &resultlen);
    
    MPI_Status status;
    
    int gpu_id_to_avoid = *gpu_id;

    clock_t clock_start;
    if (myID == (numprocs-1))
      clock_start = clock();        

    // Unless we are the first thread, wait for a message from the previous thread:
    // The MPI_Recv command will block the execution of the code until the previous threads have communicated and shared the appropriate information.
    if (0!=myID)
    {     
      MPI_Recv(previous_processor_name, NODE_NAME_LENGTH, MPI_CHAR, myID-1, 111, MPI_COMM_WORLD, &status);   // Receive the processor name and gpu_id from the previous thread
          // printf("\n -> MPI_Recv thread %d: gpu_id=%d, %s\n", myID, (int)previous_processor_name[NODE_NAME_LENGTH-1], previous_processor_name); fflush(stdout);  //!!Verbose!! 
    }
    
    // Compare the 30 first characters of the 2 names to see if we changed the node, except for the first thread that allways gets GPU 0:
    if ((0==myID) || (0!=strncmp(processor_name, previous_processor_name, NODE_NAME_LENGTH-1)))
    { 
      *gpu_id = 0;    // Thread in a new node: assign to GPU 0:
    }
    else
    {
      // Current thread in the same node as the previous one: assign next GPU (previous GPU id given in element NODE_NAME_LENGTH-1 of the array)
      *gpu_id = (int)previous_processor_name[NODE_NAME_LENGTH-1] + 1;
    }

    // Set the following GPU if this is the one to be skipped (given in the input file):
    if (*gpu_id == gpu_id_to_avoid)
    {
      *gpu_id = *gpu_id + 1;  
      printf("             Skipping GPU %d in thread %d (%s), as selected in the input file: gpu_id=%d\n", gpu_id_to_avoid, myID, processor_name, *gpu_id); fflush(stdout);
    }
    
  
    // [Would crash if all GPUs are detected connected to one monitor.]
    // //!!DeBuG!! MC-GPU_v1.4!! Skip GPUs connected to a monitor, if more GPUs available:
    // checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *gpu_id));    
    // if (0!=deviceProp.kernelExecTimeoutEnabled)                                 //!!DeBuG!! 
    // {
    //   if((*gpu_id)<(deviceCount-1))                                             //!!DeBuG!! 
    //   {      
    //     printf("\n       ==> CUDA: GPU #%d is connected to a display and the CUDA driver would limit the kernel run time. Skipping this GPU!!\n", *gpu_id); //!!DeBuG!!
    //     *gpu_id = (*gpu_id)+1;                                                  //!!DeBuG!!
    //   }
    // }
  
       
    // Send the processor and GPU id to the following thread, unless we are the last thread:
    if (myID != (numprocs-1))
    { 
      processor_name[NODE_NAME_LENGTH-1] = (char)(*gpu_id);  // Store GPU number in the last element of the array
      
          // printf(" <- MPI_Send thread %d: gpu_id=%d, %s\n", myID, (int)processor_name[NODE_NAME_LENGTH-1], processor_name); fflush(stdout);  //!!Verbose!!
      MPI_Send(processor_name, NODE_NAME_LENGTH, MPI_CHAR, myID+1, 111, MPI_COMM_WORLD);  // Send processor name and gpu_id to the following thread (tag is the current thread id)
    }
    else
    {
      printf("           -- Time spent communicating between threads to determine the GPU id to use in each thread: %.6f s\n", ((double)(clock()-clock_start))/CLOCKS_PER_SEC); fflush(stdout);
    }    
  }  
#endif  


  if (*gpu_id>=deviceCount)
  {
    printf("\n\n  \033[33m!!WARNING!!\033[0m The selected GPU number is too high, this device number does not exist!! GPU_id (starting at 0)=%d, deviceCount=%d\n", (*gpu_id), deviceCount); fflush(stdout);
    if (numprocs==1)
    {
      *gpu_id = gpuGetMaxGflopsDeviceId();
      printf("            Selecting the fastest GPU available using gpuGetMaxGflopsDeviceId(): GPU_id = %d\n\n", (*gpu_id)); fflush(stdout);
    }    
    else
    {
      exit(-1);    
    }
  }     

  checkCudaErrors(cudaGetDeviceProperties(&deviceProp, *gpu_id));   // Re-load card properties in case we chaged gpu_id
  if (deviceProp.major>99 || deviceProp.minor>99)
  {
    printf("\n\n\n  \033[31m!!!!ERROR!!\033[0m The selected GPU device does not support CUDA!! GPU_id=%d, deviceCount=%d, compute capability=%d.%d\n\n\n", (*gpu_id), deviceCount, deviceProp.major,deviceProp.minor);
    exit(-1);
  }
  
  checkCudaErrors(cudaSetDevice(*gpu_id));   // Set the GPU device. (optionally use: cutGetMaxGflopsDeviceId())
        
  if (deviceProp.major>1)
  {
    
#ifdef LARGE_CACHE  
    // -- Compute capability > 1: set a large L1 cache for the global memory, reducing the size of the shared memory:
    //       cudaFuncCachePreferShared: shared memory is 48 KB
    //       cudaFuncCachePreferL1: shared memory is 16 KB
    //       cudaFuncCachePreferNone: no preference
    printf("\n       ==> CUDA: LARGE_CACHE defined --> setting a large global memory cache (L1) and a small shared memory (cudaFuncCachePreferL1).\n");
    cudaFuncSetCacheConfig(track_particles, cudaFuncCachePreferL1);            // -- Set a large cache instead of a large shared memory.
        // #else
        // -- Using default:
        // printf("\n       ==> CUDA: LARGE_CACHE not defined --> setting a large shared memory and a small global memory cache (cudaFuncCachePreferShared).\n");
        //    cudaFuncSetCacheConfig(track_particles, cudaFuncCachePreferShared);            // !!DeBuG!! Setting size of shared memory/global cache
#endif

  }

  register int GPU_cores = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;    // CUDA SDK function to get the number of GPU cores

  // -- Reading the device properties:
  
#ifdef USING_MPI   
  printf("\n       ==> CUDA (MPI process #%d): %d CUDA enabled GPU detected! Using device #%d: \"%s\"\n", myID, deviceCount, (*gpu_id), deviceProp.name);    
#else  
  printf("\n       ==> CUDA: %d CUDA enabled GPU detected! Using device #%d: \"%s\"\n", deviceCount, (*gpu_id), deviceProp.name);    
#endif
  printf("                 Compute capability: %d.%d", deviceProp.major, deviceProp.minor);
  printf("; Global memory: %.3f Mbyte", deviceProp.totalGlobalMem/(1024.f*1024.f));
  int driverVersion = 0, runtimeVersion = 0;  
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf("; CUDA Driver Version: %d.%d, Runtime Version: %d.%d\n", driverVersion/1000, driverVersion%100, runtimeVersion/1000, runtimeVersion%100);

  if (0!=deviceProp.kernelExecTimeoutEnabled)
  {
    printf("\n\n\n   \033[33m!!WARNING!!\033[0m The selected GPU is connected to a display and therefore CUDA driver will limit the kernel run time to 5 seconds and the simulation will likely fail.\n");
    // exit(-1);
  }    

  fflush(stdout);
  
  clock_t clock_init = clock();    

  // -- Allocate the constant variables in the device:
  checkCudaErrors(cudaMemcpyToSymbol(voxel_data_CONST,     voxel_data,     sizeof(struct voxel_struct)));
  checkCudaErrors(cudaMemcpyToSymbol(source_energy_data_CONST, source_energy_data, sizeof(struct source_energy_struct)));
  
// Source, detector data now copied to global memory and transfered to shared memory in the kernel. OLD CODE:  checkCudaErrors(cudaMemcpyToSymbol(detector_data_CONST,  detector_data,  sizeof(struct detector_struct)));
  
  checkCudaErrors(cudaMemcpyToSymbol(mfp_table_data_CONST, mfp_table_data, sizeof(struct linear_interp)));

  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_x_min_CONST, dose_ROI_x_min, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_x_max_CONST, dose_ROI_x_max, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_y_min_CONST, dose_ROI_y_min, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_y_max_CONST, dose_ROI_y_max, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_z_min_CONST, dose_ROI_z_min, sizeof(short int)));
  checkCudaErrors(cudaMemcpyToSymbol(dose_ROI_z_max_CONST, dose_ROI_z_max, sizeof(short int)));
  


  double total_mem = sizeof(struct voxel_struct)+sizeof(struct source_struct)+sizeof(struct detector_struct)+sizeof(struct linear_interp) + 6*sizeof(short int);  

  // -- Allocate the device global memory:

  if (*dose_ROI_x_max > -1)  // Allocate dose array only if the tally is not disabled
  {
    checkCudaErrors(cudaMalloc((void**) voxels_Edep_device, voxels_Edep_bytes));
    if (*voxels_Edep_device==NULL)
    {
      printf("\n cudaMalloc ERROR!! Error allocating the dose array on the device global memory!! (%lf Mbytes)\n", voxels_Edep_bytes/(1024.0*1024.0));
      exit(-1);
    }
  }
  
  checkCudaErrors(cudaMalloc((void**) voxel_mat_dens_device, voxel_mat_dens_bytes));
  checkCudaErrors(cudaMalloc((void**) image_device,          image_bytes));
  checkCudaErrors(cudaMalloc((void**) mfp_Woodcock_table_device, mfp_Woodcock_table_bytes));
  checkCudaErrors(cudaMalloc((void**) mfp_table_a_device,    mfp_table_bytes));
  checkCudaErrors(cudaMalloc((void**) mfp_table_b_device,    mfp_table_bytes));
  checkCudaErrors(cudaMalloc((void**) rayleigh_table_device, sizeof(struct rayleigh_struct)));
  checkCudaErrors(cudaMalloc((void**) compton_table_device,  sizeof(struct compton_struct))); 
  checkCudaErrors(cudaMalloc((void**) detector_data_device,  num_projections*sizeof(struct detector_struct)));
  checkCudaErrors(cudaMalloc((void**) source_data_device,    num_projections*sizeof(struct source_struct)));    // The array of detectors, sources has "MAX_NUM_PROJECTIONS" elements but I am allocating only the used "num_projections" elements to the GPU
  
  if (flag_material_dose==1)
    checkCudaErrors(cudaMalloc((void**) materials_dose_device, MAX_MATERIALS*sizeof(ulonglong2)));    // !!tally_materials_dose!!
  
  total_mem = voxels_Edep_bytes + voxel_mat_dens_bytes + image_bytes + mfp_Woodcock_table_bytes + 2*mfp_table_bytes + sizeof(struct compton_struct) + sizeof(struct rayleigh_struct) + num_projections*(sizeof(struct detector_struct) + sizeof(struct source_struct));
  if (*voxel_mat_dens_device==NULL || *image_device==NULL || *mfp_Woodcock_table_device==NULL || *mfp_table_a_device==NULL ||
      *mfp_table_a_device==NULL || *rayleigh_table_device==NULL || *compton_table_device==NULL || *detector_data_device==NULL || *source_data_device==NULL)
  {
    printf("\n cudaMalloc ERROR!! Device global memory not correctly allocated!! (%lf Mbytes)\n", total_mem/(1024.0*1024.0));
    exit(-1);
  }
  else
  {
    MASTER_THREAD printf("       ==> CUDA: GLOBAL memory used: %lf Mbytes (%.1lf%%)\n", total_mem/(1024.0*1024.0), 100.0*total_mem/deviceProp.totalGlobalMem);
  }

  // --Copy the host memory to the device:
  checkCudaErrors(cudaMemcpy(*voxel_mat_dens_device, voxel_mat_dens, voxel_mat_dens_bytes,                          cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*mfp_Woodcock_table_device, mfp_Woodcock_table, mfp_Woodcock_table_bytes,              cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*mfp_table_a_device,    mfp_table_a,    mfp_table_bytes,                               cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*mfp_table_b_device,    mfp_table_b,    mfp_table_bytes,                               cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*rayleigh_table_device, rayleigh_table, sizeof(struct rayleigh_struct),                cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*compton_table_device,  compton_table,  sizeof(struct compton_struct),                 cudaMemcpyHostToDevice));  
  checkCudaErrors(cudaMemcpy(*detector_data_device,  detector_data,  num_projections*sizeof(struct detector_struct),cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(*source_data_device,    source_data,    num_projections*sizeof(struct source_struct),  cudaMemcpyHostToDevice));  
  

  //   --Init the image array to 0 using a GPU kernel instead of cudaMemcpy:
  //     Simple version: checkCudaErrors( cudaMemcpy( image_device, image, image_bytes, cudaMemcpyHostToDevice) );

  int pixels_per_image = detector_data[0].num_pixels.x * detector_data[0].num_pixels.y;

  init_image_array_GPU<<<(int)(ceil(pixels_per_image/128.0f)+0.01f),128>>>(*image_device, pixels_per_image);
    cudaDeviceSynchronize();
    getLastCudaError("\n\n !!Kernel execution failed initializing the image array!! ");  // Check if kernel execution generated any error:


  //   --Init the dose array to 0 using a GPU kernel, if the tally is not disabled:
  if (*dose_ROI_x_max > -1)
  {      
    
    checkCudaErrors(cudaMemcpy(*voxels_Edep_device, voxels_Edep, voxels_Edep_bytes, cudaMemcpyHostToDevice) );

  }
  
  // Init materials_dose array in GPU with 0 (same as host):
  if (flag_material_dose==1)
    checkCudaErrors(cudaMemcpy(*materials_dose_device, materials_dose, MAX_MATERIALS*sizeof(ulonglong2), cudaMemcpyHostToDevice));   // !!tally_materials_dose!!
  
}


////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
//! Guestimate a good number of blocks to estimate the speed of different generations 
//! of GPUs. Slower GPUs will simulate less particles and hopefully the fastest GPUs 
//! will not have to wait much. If the speed is not accurately estimated in the speed test
//! some GPUs will simulate longer than others and valuable simulation time will be wasted 
//! in the idle GPUs.
//!
//! In this function the "optimum" number of blocks for the speed test is heuristically 
//! computed as the product of three GPU characteristics:
//!   [2.0] * [number of GPU cores] * [core frequency] * [major CUDA compute capability] + [100]
//!
//! The factor 2.0 is arbitrary and can be modified depending on the case (for short 
//! simulations this value may have to be reduced or the speed test will take longer 
//! than the whole simulation). The constant 100 blocks are added to try to get enough 
//! blocks for a reliable timing of slow GPUs.
//!
//! For example, an NVIDIA GeForce 290 will get:
//!   2.0 * 240 (cores) * 1.24 (GHz) * 1 (major compute capability) + 100 =  695.2 ~  695 blocks
//! An NVIDIA GeForce 580 will get:
//!   2.0 * 512 (cores) * 1.54 (GHz) * 2 (major compute capability) + 100 = 3253.9 ~ 3254 blocks 
//! In total the 580 gets 5.7 times more blocks than the 290.
//!
//!       @param[in] gpu_id   GPU number
//!       @param[out] num_blocks   Returns a number of blocks related to the expected GPU speed
////////////////////////////////////////////////////////////////////////////////
int guestimate_GPU_performance(int gpu_id)
{          
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, gpu_id);
  float num_cores       = (float) _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount;
  float comp_capability = (float) deviceProp.major;
  float frequency       = deviceProp.clockRate*1.0e-6f;
  
  return (int)(2.0f*num_cores*frequency*comp_capability + 100.0f + 0.50f);
}
  
  
#endif
////////////////////////////////////////////////////////////////////////////////




////////////////////////////////////////////////////////////////////////////////
//! Report the tallied image in ASCII and binary form (32-bit floats).
//! Separate images for primary and scatter radiation are generated.
//! 
//!
//!       @param[in] file_name_output   File where tallied image is reported
//!       @param[in] detector_data   Detector description read from the input file (pointer to detector_struct)
//!       @param[in] image  Tallied image (in meV per pixel)
//!       @param[in] time_elapsed   Time elapsed during the main loop execution (in seconds)
//!       @param[in] total_histories   Total number of x-rays simulated
////////////////////////////////////////////////////////////////////////////////
int report_image(char* file_name_output, struct detector_struct* detector_data, struct source_struct* source_data, float mean_energy_spectrum, unsigned long long int* image, double time_elapsed, unsigned long long int total_histories, int current_projection, int num_projections, double D_angle, double initial_angle, int myID, int numprocs)
{
  
  //  -Find current angle
  double current_angle = initial_angle+current_projection*D_angle;

  // -- Report data:
  printf("\n\n          *** IMAGE TALLY PERFORMANCE REPORT ***\n");
  
  if(num_projections!=1)   // Output the projection angle when simulating a CT:
  {
    printf("              CT projection %d of %d: angle from X axis = %lf \n", current_projection+1, num_projections, current_angle*RAD2DEG);
  }
  
  printf("              Simulated x rays:    %lld\n", total_histories);
  printf("              Simulation time [s]: %.2f\n", time_elapsed);
  if (time_elapsed>0.000001)
    printf("              Speed [x-rays/s]:    %.2f\n\n", ((double)total_histories)/time_elapsed);

  char file_name_buf[250];
  strncpy(file_name_buf, file_name_output, 250);
  strcat(file_name_buf,".txt");                       // !!ASCII!! 
  FILE* file_ptr = fopen(file_name_buf, "w");
  
  if (file_ptr==NULL)
  {
    printf("\n\n   !!fopen ERROR report_image!! File %s can not be opened!!\n", file_name_output);
    exit(-3);
  }
  
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "#     ***         MC-GPU, version 1.3 (http://code.google.com/p/mcgpu/)         ***\n");
  fprintf(file_ptr, "#     ***                                                                       ***\n");
  fprintf(file_ptr, "#     ***                     Andreu Badal (Andreu.Badal-Soler@fda.hhs.gov)     ***\n");
  fprintf(file_ptr, "#     *****************************************************************************\n");
  fprintf(file_ptr, "# \n");  
#ifdef USING_CUDA
  fprintf(file_ptr, "#  *** SIMULATION IN THE GPU USING CUDA ***\n");
#else
  fprintf(file_ptr, "#  *** SIMULATION IN THE CPU ***\n");
#endif  
  fprintf(file_ptr, "#\n");
  fprintf(file_ptr, "#  Image created counting the energy arriving at each pixel: ideal energy integrating detector.\n");
  fprintf(file_ptr, "#  Pixel value units: eV/cm^2 per history (energy fluence).\n");


  if(num_projections!=1)   // Output the projection angle when simulating a CT:
  {
    fprintf(file_ptr, "#  CT projection %d of %d: angle from X axis = %lf \n", current_projection+1, num_projections, current_angle*RAD2DEG);
  }  

  fprintf(file_ptr, "#  Focal spot position = (%.8f,%.8f,%.8f), cone beam direction = (%.8f,%.8f,%.8f)\n", source_data[current_projection].position.x, source_data[current_projection].position.y, source_data[current_projection].position.z, source_data[current_projection].direction.x, source_data[current_projection].direction.y, source_data[current_projection].direction.z);

  fprintf(file_ptr, "#  Pixel size:  %lf x %lf = %lf cm^2\n", 1.0/(double)(detector_data[0].inv_pixel_size_X), 1.0/(double)(detector_data[0].inv_pixel_size_Z), 1.0/(double)(detector_data[0].inv_pixel_size_X*detector_data[0].inv_pixel_size_Z));
  
  fprintf(file_ptr, "#  Number of pixels in X and Z:  %d  %d\n", detector_data[0].num_pixels.x, detector_data[0].num_pixels.y);
  fprintf(file_ptr, "#  (X rows given first, a blank line separates the different Z values)\n");
  fprintf(file_ptr, "# \n");
  fprintf(file_ptr, "#  [NON-SCATTERED] [COMPTON] [RAYLEIGH] [MULTIPLE-SCATTING]\n");
  fprintf(file_ptr, "# ==========================================================\n");

  const double SCALE = 1.0/SCALE_eV;    // conversion to eV using the inverse of the constant used in the "tally_image" kernel function (defined in the header file)
  const double NORM = SCALE * detector_data[0].inv_pixel_size_X * detector_data[0].inv_pixel_size_Z / ((double)total_histories);  // ==> [eV/cm^2 per history]
  double energy_noScatter, energy_compton, energy_rayleigh, energy_multiscatter;
  double energy_integral = 0.0;   // Integrate (add) the energy in the image pixels [meV]
  double maximum_energy_pixel = -100.0;  // Find maximum pixel signal
  int maximum_energy_pixel_x=0, maximum_energy_pixel_y=0, maximum_energy_pixel_number=0;   

  int pixels_per_image = (detector_data[0].num_pixels.x*detector_data[0].num_pixels.y), pixel=0;
  int i, j;
  for(j=0; j<detector_data[0].num_pixels.y; j++)
  {
    for(i=0; i<detector_data[0].num_pixels.x; i++)
    {
      energy_noScatter    = (double)(image[pixel]);
      energy_compton      = (double)(image[pixel +   pixels_per_image]);
      energy_rayleigh     = (double)(image[pixel + 2*pixels_per_image]);
      energy_multiscatter = (double)(image[pixel + 3*pixels_per_image]);

      // -- Write the results in an external file; the image corresponding to all particles not written: it has to be infered adding all images
      fprintf(file_ptr, "%.8lf %.8lf %.8lf %.8lf\n", NORM*energy_noScatter, NORM*energy_compton, NORM*energy_rayleigh, NORM*energy_multiscatter);
      
      register double total_energy_pixel = energy_noScatter + energy_compton + energy_rayleigh + energy_multiscatter;   // Find and report the pixel with maximum signal
      if (total_energy_pixel>maximum_energy_pixel)
      {
        maximum_energy_pixel = total_energy_pixel;
        maximum_energy_pixel_x = i;
        maximum_energy_pixel_y = j;
        maximum_energy_pixel_number = pixel;
      }            
      energy_integral += total_energy_pixel;   // Count total energy in the whole image      
      
      pixel++;   // Move to next pixel
    }
    fprintf(file_ptr, "\n");     // Separate rows with an empty line for visualization with gnuplot.
  }
  
  fprintf(file_ptr, "#   *** Simulation REPORT: ***\n");
  fprintf(file_ptr, "#       Fraction of energy detected: %.3lf%%\n", 100.0*SCALE*(energy_integral/(double)(total_histories))/(double)(mean_energy_spectrum));
  fprintf(file_ptr, "#       Maximum energy detected at: (x,y)=(%i,%i) -> pixel value = %lf eV/cm^2\n", maximum_energy_pixel_x, maximum_energy_pixel_y, NORM*maximum_energy_pixel);
  fprintf(file_ptr, "#       Simulated x rays:    %lld\n", total_histories);
  fprintf(file_ptr, "#       Simulation time [s]: %.2f\n", time_elapsed);
  if (time_elapsed>0.000001)
    fprintf(file_ptr, "#       Speed [x-rays/sec]:  %.2f\n\n", ((double)total_histories)/time_elapsed);
   
  fclose(file_ptr);  // Close output file and flush stream

  printf("              Fraction of initial energy arriving at the detector:  %.3lf%%\n", 100.0*SCALE*(energy_integral/(double)(total_histories))/(double)(mean_energy_spectrum));
  printf("              Maximum energy detected at: (x,y)=(%i,%i). Maximum pixel value = %lf eV/cm^2\n\n", maximum_energy_pixel_x, maximum_energy_pixel_y, NORM*maximum_energy_pixel);  
  fflush(stdout);
  
  
  // -- Binary output:   
  float energy_float;
  char file_binary[250];
  strncpy (file_binary, file_name_output, 250);
  strcat(file_binary,".raw");                       // !!BINARY!! 
  FILE* file_binary_ptr = fopen(file_binary, "w");  // !!BINARY!!
  if (file_binary_ptr==NULL)
  {
    printf("\n\n   !!fopen ERROR report_image!! Binary file %s can not be opened for writing!!\n", file_binary);
    exit(-3);
  }
  
  // [MCGPULite] Total.
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i] + image[i + pixels_per_image] + image[i + 2*pixels_per_image] + image[i + 3*pixels_per_image]) );  // Total image (scatter + primary)
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);   // Write pixel data in a binary file that can be easyly open in imageJ. !!BINARY!!
  }
  // [MCGPULite] Primary.
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i]) );  // Non-scattered image
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  }
  // [MCGPULite] It now outputs summed scatter image.
  for(i=0; i<pixels_per_image; i++)
  {
    energy_float = (float)( NORM * (double)(image[i + pixels_per_image] + image[i + 2*pixels_per_image] + image[i + 3*pixels_per_image]) );  // Scatter image
    fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  }
  
  // [MCGPULite] Individual scatter images are no longer output.
  // for(i=0; i<pixels_per_image; i++)
  // {
  //   energy_float = (float)( NORM * (double)(image[i + pixels_per_image]) );  // Compton image
  //   fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  // }
  // for(i=0; i<pixels_per_image; i++)
  // {
  //   energy_float = (float)( NORM * (double)(image[i + 2*pixels_per_image]) );  // Rayleigh image
  //   fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  // }
  // for(i=0; i<pixels_per_image; i++)
  // {
  //   energy_float = (float)( NORM * (double)(image[i + 3*pixels_per_image]) );  // Multiple-scatter image
  //   fwrite(&energy_float, sizeof(float), 1, file_binary_ptr);
  // }       
  
  fclose(file_binary_ptr);    
  
    
  return 0;     // Report could return not 0 to continue the simulation...
}
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//!  Sets the CT trajectory: store in memory the source and detector rotations
//!  that are needed to calculate the multiple projections.
//!  The first projection (0) was previously initialized in function "read_input".
//!  
//!
//!  ASSUMPTIONS: the CT scan plane must be perpendicular to the Z axis, ie,
//!               the initial direction of the particles must have w=0!
//!
///////////////////////////////////////////////////////////////////////////////
void set_CT_trajectory(int myID, int num_projections, double D_angle, double angularROI_0, double angularROI_1, double SRotAxisD, struct source_struct* source_data, struct detector_struct* detector_data, double vertical_translation_per_projection)
{
  MASTER_THREAD printf("\n    -- Setting the sources and detectors for the %d CT projections (MAX_NUM_PROJECTIONS=%d):\n", num_projections, MAX_NUM_PROJECTIONS);
  double cos_rX, cos_rZ, sin_rX, sin_rZ, current_angle;

  // --Set center of rotation at the input distance between source and detector:
  float3 center_rotation;
  center_rotation.x =  source_data[0].position.x + source_data[0].direction.x * SRotAxisD;
  center_rotation.y =  source_data[0].position.y + source_data[0].direction.y * SRotAxisD;
  center_rotation.z =  source_data[0].position.z;      //  + source_data[0].direction.z * SRotAxisD;   // w=0 all the time!!

  // --Angular span between projections:

  //  -Set initial angle for the source (180 degress less than the detector pointed by the direction vector; the zero angle is the X axis, increasing to +Y axis).
  current_angle = acos((double)source_data[0].direction.x);
  if (source_data[0].direction.y<0)
    current_angle = -current_angle;     // Correct for the fact that positive and negative angles have the same ACOS
  if (current_angle<0.0)
    current_angle += 2.0*PI;   // Make sure the angle is not negative, between [0,360) degrees.
  current_angle = current_angle - PI;   // Correct the fact that the source is opposite to the detector (180 degrees difference).
  if (current_angle<0.0)
    current_angle += 2.0*PI;   // Make sure the angle is not negative, between [0,360) degrees..

  MASTER_THREAD printf("         << Projection #1 >> initial_angle=%.8f , D_angle=%.8f\n", current_angle*RAD2DEG, D_angle*RAD2DEG);
  MASTER_THREAD printf("                             Source direction=(%.8f,%.8f,%.8f), position=(%.8f,%.8f,%.8f)\n", source_data[0].direction.x,source_data[0].direction.y,source_data[0].direction.z, source_data[0].position.x,source_data[0].position.y,source_data[0].position.z);

  int i;
  for (i=1; i<num_projections; i++)   // The first projection (i=0) was initialized in function "read_input".
  {
    // --Init constant parameters to the values in projection 0:
    source_data[i].cos_theta_low = source_data[0].cos_theta_low;
    source_data[i].phi_low = source_data[0].phi_low;
    source_data[i].D_cos_theta = source_data[0].D_cos_theta;
    source_data[i].D_phi = source_data[0].D_phi;
    source_data[i].max_height_at_y1cm = source_data[0].max_height_at_y1cm;    
    detector_data[i].sdd = detector_data[0].sdd;
    detector_data[i].width_X = detector_data[0].width_X;
    detector_data[i].height_Z = detector_data[0].height_Z;
    detector_data[i].inv_pixel_size_X = detector_data[0].inv_pixel_size_X;
    detector_data[i].inv_pixel_size_Z = detector_data[0].inv_pixel_size_Z;
    detector_data[i].num_pixels = detector_data[0].num_pixels;
    detector_data[i].total_num_pixels = detector_data[0].total_num_pixels;
    detector_data[i].rotation_flag = detector_data[0].rotation_flag;
        
        
    // --Set the new source location and direction, for the current CT projection:
    current_angle += D_angle;
    if (current_angle>=(2.0*PI-0.0001))
      current_angle -= 2.0*PI;   // Make sure the angle is not above or equal to 360 degrees.

    source_data[i].position.x = center_rotation.x + SRotAxisD*cos(current_angle);
    source_data[i].position.y = center_rotation.y + SRotAxisD*sin(current_angle);
    source_data[i].position.z = source_data[i-1].position.z + vertical_translation_per_projection;   //  The Z position can increase between projections for a helical scan. But rotation still around Z always: (w=0)!!

    source_data[i].direction.x = center_rotation.x - source_data[i].position.x;
    source_data[i].direction.y = center_rotation.y - source_data[i].position.y;
    source_data[i].direction.z = 0.0f;    //  center_rotation.z - source_data[0].position.z;   !! w=0 all the time!!  

    double norm = 1.0/sqrt((double)source_data[i].direction.x*(double)source_data[i].direction.x + (double)source_data[i].direction.y*(double)source_data[i].direction.y /* + source_data[i].direction.z*source_data[i].direction.z*/);
    source_data[i].direction.x = (float)(((double)source_data[i].direction.x)*norm);
    source_data[i].direction.y = (float)(((double)source_data[i].direction.y)*norm);
      // source_data[i].direction.z = (float)(((double)source_data[i].direction.z)*norm);

    // --Set the new detector in front of the new source:
    detector_data[i].center.x = source_data[i].position.x + source_data[i].direction.x * detector_data[i].sdd;   // Set the center of the detector straight ahead of the focal spot.
    detector_data[i].center.y = source_data[i].position.y + source_data[i].direction.y * detector_data[i].sdd;
    detector_data[i].center.z = source_data[i].position.z;    //  + source_data[i].direction.z * detector_data[i].sdd;   !! w=0 all the time!!

    double rotX, rotZ;

      //  detector_data[0].rotation_flag = 1;   //  Already set in read_input!

    // -- Rotate the detector center to +Y:
    //    Set the rotation that will bring particles from the detector plane to +Y=(0,+1,0) through a rotation around X and around Z (counter-clock):
    rotX = 0.0;   // !! w=0 all the time!!  CORRECT CALCULATION:  acos(source_data[0].direction.z) - 0.5*PI;  // Rotate to +Y = (0,+1,0) --> rotX_0 =  -PI/2

    if ( (source_data[i].direction.x*source_data[i].direction.x + source_data[i].direction.y*source_data[i].direction.y) > 1.0e-8 )   // == u^2+v^2 > 0
      if (source_data[i].direction.y >= 0.0f)
        rotZ = 0.5*PI - acos(source_data[i].direction.x/sqrt(source_data[i].direction.x*source_data[i].direction.x + source_data[i].direction.y*source_data[i].direction.y));
      else
        rotZ = 0.5*PI - (-acos(source_data[i].direction.x/sqrt(source_data[i].direction.x*source_data[i].direction.x + source_data[i].direction.y*source_data[i].direction.y)));
    else
      rotZ = 0.0;   // Vector pointing to +Z, do not rotate around Z then.

    MASTER_THREAD printf("         << Projection #%d >> current_angle=%.8f degrees (rotation around Z axis = %.8f)\n", (i+1), current_angle*RAD2DEG, rotZ*RAD2DEG);
    MASTER_THREAD printf("                             Source direction = (%.8f,%.8f,%.8f) , position = (%.8f,%.8f,%.8f)\n", source_data[i].direction.x,source_data[i].direction.y,source_data[i].direction.z, source_data[i].position.x,source_data[i].position.y,source_data[i].position.z);

    cos_rX = cos(rotX);
    cos_rZ = cos(rotZ);
    sin_rX = sin(rotX);
    sin_rZ = sin(rotZ);
    detector_data[i].rot_inv[0] =  cos_rZ;    // Rotation matrix RxRz:
    detector_data[i].rot_inv[1] = -sin_rZ;
    detector_data[i].rot_inv[2] =  0.0f;
    detector_data[i].rot_inv[3] =  cos_rX*sin_rZ;
    detector_data[i].rot_inv[4] =  cos_rX*cos_rZ;
    detector_data[i].rot_inv[5] = -sin_rX;
    detector_data[i].rot_inv[6] =  sin_rX*sin_rZ;
    detector_data[i].rot_inv[7] =  sin_rX*cos_rZ;
    detector_data[i].rot_inv[8] =  cos_rX;


    detector_data[i].corner_min_rotated_to_Y.x = detector_data[i].center.x*detector_data[i].rot_inv[0] + detector_data[i].center.y*detector_data[i].rot_inv[1] + detector_data[i].center.z*detector_data[i].rot_inv[2];
    detector_data[i].corner_min_rotated_to_Y.y = detector_data[i].center.x*detector_data[i].rot_inv[3] + detector_data[i].center.y*detector_data[i].rot_inv[4] + detector_data[i].center.z*detector_data[i].rot_inv[5];
    detector_data[i].corner_min_rotated_to_Y.z = detector_data[i].center.x*detector_data[i].rot_inv[6] + detector_data[i].center.y*detector_data[i].rot_inv[7] + detector_data[i].center.z*detector_data[i].rot_inv[8];

    // -- Set the lower corner (minimum) coordinates at the normalized orientation: +Y. The detector has thickness 0.
    detector_data[i].corner_min_rotated_to_Y.x = detector_data[i].corner_min_rotated_to_Y.x - 0.5*detector_data[i].width_X;
//  detector_data[i].corner_min_rotated_to_Y.y = detector_data[i].corner_min_rotated_to_Y.y;
    detector_data[i].corner_min_rotated_to_Y.z = detector_data[i].corner_min_rotated_to_Y.z - 0.5*detector_data[i].height_Z;

    // *** Init the fan beam source model:

      rotZ = -rotZ;   // The source rotation is the inverse of the detector.
      cos_rX = cos(rotX);
      cos_rZ = cos(rotZ);
      sin_rX = sin(rotX);
      sin_rZ = sin(rotZ);
      // --Rotation around X (alpha) and then around Z (phi): Rz*Rx (oposite of detector rotation)
      source_data[i].rot_fan[0] =  cos_rZ;
      source_data[i].rot_fan[1] = -cos_rX*sin_rZ;
      source_data[i].rot_fan[2] =  sin_rX*sin_rZ;
      source_data[i].rot_fan[3] =  sin_rZ;
      source_data[i].rot_fan[4] =  cos_rX*cos_rZ;
      source_data[i].rot_fan[5] = -sin_rX*cos_rZ;
      source_data[i].rot_fan[6] =  0.0f;
      source_data[i].rot_fan[7] =  sin_rX;
      source_data[i].rot_fan[8] =  cos_rX;
  }
}


///////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Initialize the first seed of the pseudo-random number generator (PRNG) 
//! RANECU to a position far away from the previous history (leap frog technique).
//! This function is equivalent to "init_PRNG" but only updates one of the seeds.
//!
//! Note that if we use the same seed number to initialize the 2 MLCGs of the PRNG
//! we can only warranty that the first MLCG will be uncorrelated for each value
//! generated by "update_seed_PRNG". There is a tiny chance that the final PRNs will
//! be correlated because the leap frog on the first MLCG will probably go over the
//! repetition cycle of the MLCG, which is much smaller than the full RANECU. But any
//! correlataion is extremely unlikely. Function "init_PRNG" doesn't have this issue.
//!
//!       @param[in] batch_number   Elements to skip (eg, MPI thread_number).
//!       @param[in] total_histories   Histories to skip.
//!       @param[in,out] seed   Initial PRNG seeds; returns the updated seed.
////////////////////////////////////////////////////////////////////////////////
inline void update_seed_PRNG(int batch_number, unsigned long long int total_histories, int* seed)
{
  if (0==batch_number)
    return;
    
  unsigned long long int leap = total_histories * (batch_number * LEAP_DISTANCE);
  int y = 1;
  int z = a1_RANECU;
  // -- Calculate the modulo power '(a^leap)MOD(m)' using a divide-and-conquer algorithm adapted to modulo arithmetic
  for(;;)
  {
    // (A2) Halve n, and store the integer part and the residue
    if (0!=(leap&01))  // (bit-wise operation for MOD(leap,2), or leap%2 ==> proceed if leap is an odd number)  Equivalent: t=(short)(leap%2);
    {
      leap >>= 1;     // Halve n moving the bits 1 position right. Equivalent to:  leap=(leap/2);  
      y = abMODm(m1_RANECU,z,y);      // (A3) Multiply y by z:  y = [z*y] MOD m
      if (0==leap) break;         // (A4) leap==0? ==> finish
    }
    else           // (leap is even)
    {
      leap>>= 1;     // Halve leap moving the bits 1 position right. Equivalent to:  leap=(leap/2);
    }
    z = abMODm(m1_RANECU,z,z);        // (A5) Square z:  z = [z*z] MOD m
  }
  // AjMODm1 = y;                 // Exponentiation finished:  AjMODm = expMOD = y = a^j
  // -- Compute and display the seeds S(i+j), from the present seed S(i), using the previously calculated value of (a^j)MOD(m):
  //         S(i+j) = [(a**j MOD m)*S(i)] MOD m
  //         S_i = abMODm(m,S_i,AjMODm)
  *seed = abMODm(m1_RANECU, *seed, y);
}


///////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
//! Read the energy spectrum file and initialize the Walker aliasing sampling.
//!
//!       @param[in] file_name_espc   File containing the energy spectrum (lower energy value in each bin and its emission probability).
//!       @param[in,out] source_energy_data   Energy spectrum and other source data. The Walker alias and cutoffs are initialized in this function.
//!       @param[out] mean_energy_spectrum   Mean energy in the input x-ray energy spectrum.
////////////////////////////////////////////////////////////////////////////////
void init_energy_spectrum(char* file_name_espc, struct source_energy_struct* source_energy_data, float *mean_energy_spectrum)
{
  char *new_line_ptr = NULL, new_line[250];    
  float lower_energy_bin, prob;
  float prob_espc_bin[MAX_ENERGY_BINS];    // The input probabilities of each energy bin will be discarded after Walker is initialized

  // -- Read spectrum from file:
  FILE* file_ptr = fopen(file_name_espc, "r");
  if (NULL==file_ptr)
  {
    printf("\n\n   !!init_energy_spectrum ERROR!! Error trying to read the energy spectrum input file \"%s\".\n\n", file_name_espc);
    exit(-1);
  }
  
  int current_bin = -1;
  do 
  {
    current_bin++;  // Update bin counter
    
    if (current_bin >= MAX_ENERGY_BINS)
    {
      printf("\n !!init_energy_spectrum ERROR!!: too many energy bins in the input spectrum. Increase the value of MAX_ENERGY_BINS=%d.\n", MAX_ENERGY_BINS);
      printf(  "            A negative probability marks the end of the spectrum.\n\n");
      exit(-1);
    }

    new_line_ptr = fgets_trimmed(new_line, 250, file_ptr);   // Read the following line of text skipping comments and extra spaces
    
    if (new_line_ptr==NULL)
    {
      printf("\n\n   !!init_energy_spectrum ERROR!! The input file for the x ray spectrum (%s) is not readable or incomplete (a negative probability marks the end of the spectrum).\n", file_name_espc);
      exit(-1);
    }
    
    prob = -123456789.0f;  
    
    sscanf(new_line, "%f %f", &lower_energy_bin, &prob);     // Extract the lowest energy in the bin and the corresponding emission probability from the line read 
            
    prob_espc_bin[current_bin]     = prob;
    source_energy_data->espc[current_bin] = lower_energy_bin;           
    
    if (prob == -123456789.0f)
    {
      printf("\n !!init_energy_spectrum ERROR!!: invalid energy bin number %d?\n\n", current_bin);
      exit(-1);
    }
    else if (lower_energy_bin < source_energy_data->espc[max_value(current_bin-1,0)])    // (Avoid a negative index using the macro "max_value" defined in the header file)
    {
      printf("\n !!init_energy_spectrum ERROR!!: input energy bins with decreasing energy? espc(%d)=%f, espc(%d)=%f\n\n", current_bin-1, source_energy_data->espc[max_value(current_bin-1,0)], current_bin, lower_energy_bin);
      exit(-1);
    }
    
  } 
  while (prob > -1.0e-11f);     // A negative probability marks the end of the spectrum


  // Store the number of bins read from the input energy spectrum file:
  source_energy_data->num_bins_espc = current_bin;


  // Init the remaining bins (which will not be used) with the last energy read (will be assumed as the highest energy in the last bin) and 0 probability of emission.
  register int i;
  for (i=current_bin; i<MAX_ENERGY_BINS; i++)
  {
    source_energy_data->espc[i] = lower_energy_bin;
    prob_espc_bin[i]     = 0.0f;
  }


  // Compute the mean energy in the spectrum, taking into account the energy and prob of each bin:
  float all_energy = 0.0f;
  float all_prob = 0.0f;
  for(i=0; i<source_energy_data->num_bins_espc; i++)
  {
    all_energy += 0.5f*(source_energy_data->espc[i]+source_energy_data->espc[i+1])*prob_espc_bin[i];
    all_prob   += prob_espc_bin[i];
  }  
  *mean_energy_spectrum = all_energy/all_prob;
  
          
// -- Init the Walker aliasing sampling method (as it is done in PENELOPE):
  IRND0(prob_espc_bin, source_energy_data->espc_cutoff, source_energy_data->espc_alias, source_energy_data->num_bins_espc);   //!!Walker!! Calling PENELOPE's function to init the Walker method
       
}       
      
//********************************************************************
//!    Finds the interval (x(i),x(i+1)] containing the input value    
//!    using Walker's aliasing method.                                
//!                                                                   
//!    Input:                                                         
//!      cutoff(1..n) -> interval cutoff values for the Walker method 
//!      cutoff(1..n) -> alias for the upper part of each interval    
//!      randno       -> point to be located                          
//!      n            -> no. of data points                           
//!    Output:                                                        
//!      index i of the semiopen interval where randno lies           
//!    Comments:                                                      
//!      -> The cutoff and alias values have to be previously         
//!         initialised calling the penelope subroutine IRND0.        
//!                                                                   
//!                                                                   
//!    Algorithm implementation based on the PENELOPE code developed   
//!    by Francesc Salvat at the University of Barcelona. For more     
//!    info: www.oecd-nea.org/science/pubs/2009/nea6416-penelope.pdf  
//!                                                                   
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//C  PENELOPE/PENGEOM (version 2006)                                     C
//C  Copyright (c) 2001-2006                                             C
//C  Universitat de Barcelona                                            C
//C                                                                      C
//C  Permission to use, copy, modify, distribute and sell this software  C
//C  and its documentation for any purpose is hereby granted without     C
//C  fee, provided that the above copyright notice appears in all        C
//C  copies and that both that copyright notice and this permission      C
//C  notice appear in all supporting documentation. The Universitat de   C
//C  Barcelona makes no representations about the suitability of this    C
//C  software for any purpose. It is provided "as is" without express    C
//C  or implied warranty.                                                C
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
inline int seeki_walker(float *cutoff, short int *alias, float randno, int n)
{
   float RN = randno * n;                         // Find initial interval (array starting at 0):   
   int int_part = (int)(RN);                      //   -- Integer part
   float fraction_part = RN - ((float)int_part);  //   -- Fractional part

   if (fraction_part < cutoff[int_part])          // Check if we are in the aliased part
      return int_part;                            // Below the cutoff: return current value
   else
      return (int)alias[int_part];                // Above the cutoff: return alias
}     

//****************************************************************** *
//*                    SUBROUTINE IRND0                              *
//********************************************************************
//*                                                                   
//!  Initialisation of Walker's aliasing algorithm for random         
//!  sampling from discrete probability distributions.                
//!                                                                   
//! Input arguments:                                                  
//!   N ........ number of different values of the random variable.   
//!   W(1:N) ... corresponding point probabilities (not necessarily   
//!              normalised to unity).                                
//! Output arguments:                                                 
//!   F(1:N) ... cutoff values.                                       
//!   K(1:N) ... alias values.                                        
//!                                                                   
//!                                                                   
//!  This subroutine is part of the PENELOPE 2006 code developed      
//!  by Francesc Salvat at the University of Barcelona. For more       
//!  info: www.oecd-nea.org/science/pubs/2009/nea6416-penelope.pdf    
//*                                                                   
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
//C  PENELOPE/PENGEOM (version 2006)                                     C
//C  Copyright (c) 2001-2006                                             C
//C  Universitat de Barcelona                                            C
//C                                                                      C
//C  Permission to use, copy, modify, distribute and sell this software  C
//C  and its documentation for any purpose is hereby granted without     C
//C  fee, provided that the above copyright notice appears in all        C
//C  copies and that both that copyright notice and this permission      C
//C  notice appear in all supporting documentation. The Universitat de   C
//C  Barcelona makes no representations about the suitability of this    C
//C  software for any purpose. It is provided "as is" without express    C
//C  or implied warranty.                                                C
//CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC
void IRND0(float *W, float *F, short int *K, int N)
{
   register int I;
  
   //  ****  Renormalisation.
   double WS=0.0;
   for (I=0; I<N; I++)
   {   
      if(W[I] < 0.0f) 
      {
         printf("\n\n \033[31m!!!!ERROR!!\033[0m IRND0: Walker sampling initialization. Negative point probability? W(%d)=%f\n\n", I, W[I]);
         exit(-1);
      }
      WS = WS + W[I];
   }
   WS = ((double)N) / WS; 
  
   for (I=0; I<N; I++)
   {
      K[I] = I;
      F[I] = W[I] * WS;
   }
    
   if (N==1) 
      return;
     
   //  ****  Cutoff and alias values.
   float HLOW, HIGH;
   int   ILOW, IHIGH, J;
   for (I=0; I<N-1; I++)
   {
      HLOW = 1.0f;
      HIGH = 1.0f;
      ILOW = -1;
      IHIGH= -1;
      for (J=0; J<N; J++)
      {
         if(K[J]==J)
         {
            if(F[J]<HLOW)
            {
               HLOW = F[J];
               ILOW = J;
            }
            else if(F[J]>HIGH)
            {
               HIGH  = F[J];
               IHIGH = J;
            }
         }
      }
      
      if((ILOW==-1) || (IHIGH==-1)) 
        return;

      K[ILOW] = IHIGH;
      F[IHIGH]= HIGH + HLOW - 1.0f;
   }
   return;
}

///////////////////////////////////////////////////////////////////////////////
