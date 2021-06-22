from pandas.plotting import register_matplotlib_converters
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
import tensorflow as tf
register_matplotlib_converters()

PRINT_EPOCH = 3000

def softmax(action_q_vals):
    rate_a=action_q_vals / np.sum(action_q_vals)
    exp_a=np.exp(rate_a)
    soft_sum=np.sum(exp_a)
    soft=exp_a/soft_sum
    return soft

def do_action(action_list, action, action_q_vals, budget, num_stocks_list, stock_price_list):
    # TODO : apply 10,000,000 won purchase constraints
    # TODO: define action's operation=
    budget += sum(stock_price_list * num_stocks_list)
    num_stocks_list = [0] * (len(action_list) - 1)
    budget_list=[]
    
    #Criteria reward
    budget_c= budget
    num_stocks_list_c = [0] * (len(action_list) - 1)
    budget_c_c=(budget_c/(len(action_list)-1))
    for k in range(len(action_list)-1):
        n_buy_c =  budget_c_c // stock_price_list[k]
        num_stocks_list_c[k] = num_stocks_list_c[k]+n_buy_c
        budget_c -=num_stocks_list_c[k]*stock_price_list[k]
    
    port_rate=softmax(action_q_vals)
    
    for i in range(len(port_rate)):
        budget_list.append(port_rate[i]*budget)
    for k in range(len(action)):
        if action[k] != 'not_buying': # Buy : buy certain stock below 10**7 won, and sell next day
            n_buy = min(budget_list[k] // stock_price_list[action_list.index(action[k])], np.floor(10.**7/stock_price_list[action_list.index(action[k])]))
            num_stocks_list[action_list.index(action[k])] = num_stocks_list[action_list.index(action[k])]+n_buy
            #print("n_buy:",n_buy)
            #print("price:",stock_price_list[action_list.index(action[k])])
            #print(action[k],stock_price_list[action_list.index(action[k])] * n_buy)
            budget -= stock_price_list[action_list.index(action[k])] * n_buy
        else :
            pass
    return budget, num_stocks_list, port_rate, budget_c, num_stocks_list_c

def run_simulation(policy, initial_budget, initial_num_stocks, open_prices, close_prices, features):
    action_count = [0] * len(policy.actions)
    action_seq = list()

    budget = initial_budget
    num_stocks_list = initial_num_stocks
    action = 'not_buying'

    for t in range(len(open_prices)-1):
        #print(t)
        ##### TODO: define current state
        current_state = np.asmatrix(np.hstack((features[t])))
        # calculate current portfolio value
        current_portfolio = budget + sum(num_stocks_list * open_prices[t])

        ##### select action & update portfolio values
        action,action_q_vals = policy.select_action(current_state, True)
        action_seq.append(action)
        for i in range(len(action)):
            action_count[policy.actions.index(action[i])] += 1

        budget, num_stocks_list, port_rate, budget_c, num_stocks_list_c  = do_action(policy.actions, action, action_q_vals, budget, num_stocks_list, open_prices[t])

        ##### TODO: define reward
        # calculate new portofolio after taking action
        new_portfolio = budget + sum(num_stocks_list * close_prices[t])
        new_portfolio_c = budget_c + sum(num_stocks_list_c * close_prices[t])
        # calculate reward from taking an action at a state
        criteria=new_portfolio_c-current_portfolio
        reward=num_stocks_list*(close_prices[t]-open_prices[t])
        ##### TODO: define next state
        next_state = np.asmatrix(np.hstack((features[t+1])))
        action_q_vals = policy.sess.run(policy.q, feed_dict={policy.x: current_state})
        ##### update the policy after experiencing a new action
        for i in range(len(action)):
            if action[i]=='not_buying':
                action_q_vals= policy.update_q(current_state, action[i], 0, next_state, action_q_vals, port_rate[i], current_portfolio, criteria)
            else:
                action_q_vals=policy.update_q(current_state, action[i], reward[policy.actions.index(action[i])], next_state, action_q_vals,  port_rate[i], current_portfolio, criteria)
        action_q_vals = np.squeeze(np.asarray(action_q_vals))
        policy.sess.run(policy.train_op, feed_dict={policy.x: current_state, policy.y: action_q_vals})
        
        '''
        print(policy.sess.run(policy.q, feed_dict={policy.x: current_state}))
        '''
    # compute final portfolio worth
    portfolio = budget + sum(num_stocks_list * close_prices[-1])
    print('budget: {}, shares: {}, stock price: {} =>  portfolio: {}'.format(budget, num_stocks_list,
                                                                             close_prices[-1], portfolio))

    return portfolio, action_count, np.asarray(action_seq)





def run_simulations(company_list, policy, budget, num_stocks, open_prices, close_prices, features, num_epoch):
    best_portfolio = 0
    final_portfolios = list()
    PF=[]
    print("open_prices", open_prices)
    for epoch in range(num_epoch):
        print("-------- simulation {} --------".format(epoch + 1))
        policy.epsilon=policy.epsilon-0.95/num_epoch
        final_portfolio, action_count, action_seq = \
            run_simulation(policy, budget, num_stocks, open_prices, close_prices, features)
        final_portfolios.append(final_portfolio)
        print(sum(final_portfolios)/len(final_portfolios))
        print('actions : ', *zip(policy.actions, action_count), )

        if (epoch + 1) % PRINT_EPOCH == 0:
            #action_seq2 = np.concatenate([['.'], action_seq[:-1]])
            #policy.save_model("LFD_Project4_team03")
            #####
            action_seq2 = np.concatenate([action_seq, ['.']])
            #####
            for i, a in enumerate(policy.actions[:-1]):
                plt.figure(figsize=(40, 20))
                plt.title('Company {} / Epoch {}'.format(a, epoch + 1))
                plt.plot(open_prices[0: len(action_seq),i], 'grey')
                #hold on

                for k, daily_act in enumerate(action_seq):
                    if a in daily_act:

                        plt.plot(k+1, open_prices[: len(action_seq2), i][k+1], 'ro', markersize=1)
                        plt.plot(k, open_prices[: len(action_seq2), i][k], 'bo', markersize=1)

                #
                #plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq2 == a], 'ro', markersize=1) # sell
                #plt.plot(pd.DataFrame(open_prices[: len(action_seq), i])[action_seq == a], 'bo', markersize=1)  # buy
                plt.show()

        PF.append(final_portfolio)
    print(num_epoch, PF)
    plt.plot([p for p in range(num_epoch)], PF)
    plt.show()

        # ##### save if best portfolio value is updated
        # if best_portfolio < final_portfolio:
        #     best_portfolio = final_portfolio
        #     policy.save_model("LFD_project4-{}-e{}-step{}".format(company_list, num_epoch, epoch))

    print(final_portfolios[-1])