using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using NumSharp;
using System;

public class CEM{
    public int in_size = 5;    //Number of observations (pos_x, pos_y, theta, goal_x, goal_y)
    public int out_size = 2;  //Number of o

    //Policy parameters
    public int hidden_size = 5;    // How many values in the hidden layer
    public int evalation_samples = 1; // How many samples to take when evaluating a network

    //Training parameters
    public int cem_iterations = 100;    // How many total CEM iterations 
    public int cem_batch_size = 50;     // How many guassian samples in each CEM iteration
    public float cem_elite_frac = 0.5f;    // What percentage of cem samples are used to fit the guassian for next iteration
    public float cem_init_stddev = 1.0f;   // Initial CEM guassian uncertainty
    public float cem_noise_factor = 1.0f;    // Scaling factor of how much extra noise to add each iteration (noise_factor/iteration_number noise is added to std.dev.)
    public float cem_print_rate = 5;

    // Simulation paramters
    public float dt = 0.1f;    //seconds
    public int runtime = 8; //seconds

    //Car dynamics paramters
    public int v_max = 1;  //units/sec
    public float omega_max = 3.14f; //pi radians/sec = 180 deg/sec turn speed
    public float delta_max = 30.0f;

    //Car shape
    public float car_w = 5, car_l = 10;
    public const float Deg2Rad = 0.0174532924F;
    public const float Rad2Deg = 57.29578F;

    //Target task
    NDArray car_start = np.array(new double[]{0, 0, 0});
    NDArray car_goal = np.array(new double[]{50, 0});

    public NDArray two_layer_model(NDArray param, NDArray in_data) {
        //place input data in a column vector
        var in_vec = in_data.reshape(in_size, 1);

        //Layer 1 (input -> hidden)
        var m1_end = hidden_size * in_size;
        var matrix1 = np.reshape(param[new Slice(0, m1_end)], (hidden_size, in_size));
        var biases1 = np.reshape(param[new Slice(m1_end, m1_end + hidden_size)], (hidden_size, 1));
        var hidden_out = np.matmul(matrix1, in_vec) + biases1;

        for(int i = 0; i < hidden_out.shape[0]; i++){
            var tmp = hidden_out.GetDouble(i);
            if(tmp < 0){
                hidden_out[i] *= 0.1;
            }
        }
        
        //hidden_out = np.matmul(hidden_out, (hidden_out > 0)) + 0.1 * np.matmul(hidden_out, (hidden_out < 0)); //Leaky ReLU

        //Layer 2 (hiden -> output);
        var m2_start = m1_end + hidden_size;
        var m2_end = m2_start + out_size * hidden_size;
        var matrix2 = np.reshape(param[new Slice(m2_start, m2_end)], (out_size, hidden_size));
        var biases2 = np.reshape(param[new Slice(m2_end, m2_end + out_size)], (out_size, 1));
        var result = np.matmul(matrix2, hidden_out) + biases2;
        result = result.reshape(out_size);
        return result;
    }
    public Tuple<NDArray, NDArray> cem(NDArray th_mean,int n_iter,int batch_size,float elite_frac,double initial_std = 1.0) {
        var n_elite = Math.Round(batch_size * elite_frac, 0);
        var th_std = np.ones_like(th_mean) * initial_std;
        NDArray ths, ys = np.zeros(batch_size);
        for (int iter = 0; iter < n_iter; iter++){
            ths = th_std * np.random.randn(batch_size, th_mean.size) + th_mean;
            for(int i = 0; i< ths.shape[0]; i++){
                ys[i] = reward(ths[i], evalation_samples);
            }
            var elite_inds = ys.argsort<double>()["::-1"][new Slice(0, (int)n_elite)];
            var elite_ths = ths[elite_inds];
            th_mean = elite_ths.mean(axis: 0);
            th_std = elite_ths.std(axis: 0);
            th_std += cem_noise_factor / (iter + 1);
            if((iter + 1) % cem_print_rate == 0 || iter == 0){
                Console.WriteLine("Iter: " + (iter + 1).ToString() + " mean(reward)" + ys.mean().ToString() + " f(theta_mean)" + reward(th_mean, 100).ToString() + " avg(std_dev):" + th_std.mean().ToString());
            }
        }

        return new Tuple<NDArray, NDArray>(th_mean, th_std);
    }

    public Tuple<double[], NDArray> update_state(NDArray param, NDArray cur_state, NDArray goal){
        double cx = cur_state[0], cy = cur_state[1];
        double theta = cur_state[2];
        double gx = goal[0], gy = goal[1];
        NDArray action = two_layer_model(param, np.array(new double[]{cx, cy, theta, gx, gy}));
        var v = action[0];
        var omega = action[1];

        v = np.clip(v,-v_max,v_max);
        omega = np.clip(omega, -omega_max, omega_max);

        var vx = v * np.sin(theta);
        var vy = v * np.cos(theta);

        theta += omega * dt;
        cx += vx * dt;
        cy += vy *dt;
        double[] res = {cx, cy, theta};

        return new Tuple<double[], NDArray>(res, action);
    }
    public Tuple<List<NDArray>, List<NDArray>> run_model(NDArray param, NDArray init_state, NDArray goal_pos){
        List<NDArray> state_list = new List<NDArray>();
        List<NDArray> action_list = new List<NDArray>();
        var cur_state = init_state;
        state_list.Add(cur_state);
        var sim_time = runtime;
        // var reward = 0;
        for(int i = 0; i< (int)(sim_time/dt); i++){
            var tmpState = update_state(param, cur_state, goal_pos);
            cur_state = tmpState.Item1;
            state_list.Add(tmpState.Item1);
            action_list.Add(tmpState.Item2);
        }
        return new Tuple<List<NDArray>, List<NDArray>>(state_list, action_list);
    }

    public double reward(NDArray policy, int num_tasks = 10){
        double total_reward = 0;
        for(int iter =0; iter<num_tasks; iter++){
            var init_state = car_start;
            var goal_pos = car_goal;
            var tmpStateActions = run_model(policy, init_state, goal_pos);
            double task_reward = 0;
            var states = tmpStateActions.Item1;
            var actions = tmpStateActions.Item2;
            double dist = 0;
            for(int i = 0; i < actions.Count; i++){
                var cur_state = states[i + 1];
                var cur_action = actions[i];
                var dx = cur_state[0] - goal_pos[0];
                var dy = cur_state[1] - goal_pos[1];
                dist = Math.Sqrt(dx*dx + dy*dy);
                task_reward -= dist;
                task_reward -= 1 * Math.Abs(cur_action.GetDouble(0));
                task_reward -= 1 * Math.Abs(cur_action.GetDouble(1));
            }
            var final_state = states[states.Count - 1];
            var final_action = actions[actions.Count - 1];
            var final_dist = dist;
            if(final_dist < 20){
                task_reward += 1000;
            }
            if(final_dist < 10 && Math.Abs(final_action.GetDouble(0)) < 5){
                task_reward += 10000;
            }
            total_reward += task_reward;
        }
        return total_reward/ num_tasks;
    }

    public void train(){
        var policy_size = (in_size+1)*hidden_size + (hidden_size+1)*out_size;
        var init_params = np.zeros(policy_size);
        var resCem = cem(init_params, cem_iterations, cem_batch_size, cem_elite_frac, cem_init_stddev);

        var mean_policy = resCem.Item1;
        var resRun = run_model(mean_policy, car_start,car_goal);
        foreach(var s in resRun.Item2){
            Console.WriteLine(s.ToString());
        }
    }
}

public class Car : MonoBehaviour
{
    public Transform mGoalTransform;
    public Transform mCarTransform;
    public float mMaxSpeed = 30f;
    public float kRotateSpeed = 20f;
    public float kMaxWheelRotation = 30f;
    float beta, theta, delta, dtheta, dx, dz;
    float carDeg;
    float acc, vel, fraction;
    Rigidbody mRigidBody;
    Vector3 mCarParams, mCarPos;
    CEM controller;
    NDArray meanPolicy;
    private void Awake() {
        mCarTransform = GetComponent<Transform>();
        mRigidBody = GetComponent<Rigidbody>();
        mCarParams = GetComponent<BoxCollider>().size;
        acc = 10;
        vel = 0;
        delta = 0;
        theta = Mathf.Atan2(mCarTransform.forward.z, mCarTransform.forward.x);
        controller = new CEM();
        controller.car_l = mCarParams.z;
        controller.dt = Time.deltaTime;
        controller.v_max = 10;
        meanPolicy = np.load("weight.npy");
    }

    void Update() {
        NDArray curState = np.array(new double[]{mCarTransform.position.x, mCarTransform.position.z, mCarTransform.localEulerAngles.y * Mathf.Deg2Rad});
        NDArray goal = np.array(new double[]{mGoalTransform.position.x, mGoalTransform.position.z});
        var resStateAction = controller.update_state(meanPolicy, curState, goal);
        var state = resStateAction.Item1;
        mCarTransform.position = new Vector3((float)state[0], 0, (float)state[1]);
        mCarTransform.localEulerAngles = new Vector3(0, (float)state[2] * Mathf.Rad2Deg, 0);
    }

    // void Update()
    // {
    //     if (Input.GetKey(KeyCode.A))
    //     {
    //         delta -= kRotateSpeed * Time.deltaTime;
    //         if(delta < -kMaxWheelRotation)
    //             delta = -kMaxWheelRotation;
    //     }
    //     if (Input.GetKey(KeyCode.D))
    //     {
    //         delta += kRotateSpeed * Time.deltaTime;
    //         if(delta > kMaxWheelRotation)
    //             delta = kMaxWheelRotation;
    //     }
    //     if (Input.GetKey(KeyCode.W))
    //     {
    //         vel += acc * Time.deltaTime;
    //         if(vel > mMaxSpeed)
    //             vel = mMaxSpeed;
    //     }
    //     if (Input.GetKey(KeyCode.S))
    //     {
    //         vel -= acc * Time.deltaTime;
    //         if(vel < -mMaxSpeed)
    //             vel = -mMaxSpeed;
    //     }
    //     if(Input.GetKeyUp(KeyCode.A) || Input.GetKeyUp(KeyCode.D)){
    //         delta = 0;
    //     }
    //     delta *= Mathf.Deg2Rad;
    //     beta = Mathf.Atan(Mathf.Tan(delta) / 2);
    //     dx = vel * Mathf.Cos(theta + beta);
    //     dz = vel * Mathf.Sin(theta + beta);
    //     dtheta = vel * Mathf.Cos(beta) * Mathf.Tan(delta) / mCarParams.z * Mathf.Rad2Deg;
    //     mCarTransform.Translate(new Vector3(dx, 0, dz) * Time.deltaTime);
    //     mCarTransform.localEulerAngles = new Vector3(0, mCarTransform.localEulerAngles.y + dtheta* Time.deltaTime, 0);
    //     delta *= Mathf.Rad2Deg;
    // }
    private void OnCollisionEnter(Collision other) {
        vel = 0;
        //controller.v_max = 0;
    }
}
