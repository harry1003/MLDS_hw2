def train(model, opt, train_data_loader, test_data_loader, device, epochs, batch_size=100, teach_init_rate=1):
    model.train()
    crit = LanguageModelCriterion()
    for i in range(epochs):
        print("epochs:", i)
        # to see the training sentence
        loss_ep, pre_sent_e, ans1, ans2 = 0, 0, 0, 0
        # start training
        for batch in range(train_data_loader.data_size//batch_size):
            # load data 
            data, target, mask = train_data_loader.load_on_batch(
                batch * batch_size, (batch + 1) * batch_size,
                i % used_sent)
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            
            # get prediction
            pre_prob, pre_sent = model(data, target, device, teach_init_rate)
            
            # get loss
            loss = crit(pre_prob, target, mask)
            
            # backward
            opt.zero_grad()
            loss.backward()
            
            # grad clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            
            # step
            opt.step()
            
            # showing things
            loss_ep += loss
            ans1 = target[0].cpu().numpy()
            ans2 = target[1].cpu().numpy()
            pre_sent_e = pre_sent
            
        # showing things
        av_loss = loss_ep/(train_data_loader.data_size//batch_size)
        print("train loss:", av_loss)

        train_sent = train_data_loader.turn_tar_to_sent(pre_sent_e[:, 0])
        train_sent2 = train_data_loader.turn_tar_to_sent(pre_sent_e[:, 1])
        train_ans1 = train_data_loader.turn_vec_to_sent(ans1)
        train_ans2 = train_data_loader.turn_vec_to_sent(ans2)
        print()
        print("train_sent:", train_sent)
        print("ans:", train_ans1)
        print("train_sent2:", train_sent2)
        print("ans2:", train_ans2)
        print()
        validation(model, test_data_loader, device, i)

        
def validation(model, test_data_loader, device, i, teach_init_rate=1):
    model.eval()
    # load data
    data, target, mask = test_data_loader.load_on_batch(
                0, 100, i % used_sent)
    data = data.to(device)
    dummy1 = torch.zeros(target.shape).to(device)
    
    # predict
    pre_prob, pre_sent = model(data, dummy1, device, teach_init_rate)

    # print
    ans1 = target[0].cpu().numpy()
    ans2 = target[1].cpu().numpy()
    test_ans1 = test_data_loader.turn_vec_to_sent(ans1)
    test_ans2 = test_data_loader.turn_vec_to_sent(ans2)
    test_sent1 = test_data_loader.turn_tar_to_sent(pre_sent[:, 0])
    test_sent2 = test_data_loader.turn_tar_to_sent(pre_sent[:, 1])
    print("test_sent:", test_sent1)
    print("ans:", test_ans1)
    print("test_sent2:", test_sent2)
    print("ans2:", test_ans2)
    print()
    print()
    model.train()



            

