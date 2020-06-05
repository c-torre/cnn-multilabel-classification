
model_ppl = PeopleNet()
model_ppl.cuda()

criterion_ppl = nn.MSELoss(reduction="none")
# optimizer_ppl = optim.SGD(model.parameters(),lr=0.001, momentum = 0.9)
optimizer_ppl = optim.Adam(model_ppl.parameters(), lr=0.001)


# %%
epochmax = 10

losses = []
for epoch in range(epochmax):
    running_loss = 0.0
    epoch_size = 0
    for i, data in enumerate(people_loader, 0):
        # input
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()

        optimizer_ppl.zero_grad()
        outputs = model_ppl(inputs)
        # print(outputs)
        # print(labels)
        loss = criterion_ppl(outputs, labels)
        loss.backward()
        optimizer_ppl.step()

        running_loss += loss.item()
        if i % 30 == 29:
            print("[%d,%5d] loss %.3f" % (epoch + 1, i + 1, running_loss / 30))
            losses += [running_loss]
            running_loss = 0.0

print("Finished training")

# %% [markdown]
# ## Save your model
#
# It might be useful to save your model if you want to continue your work later, or use it for inference later.

# %%
torch.save(model.state_dict(), "lauritasbrain.pkl")


# %%
model = NetNoPeople()
model.load_state_dict(torch.load("NoPeopleClassifier_v1_20e.pkl"))
model.eval()
model.cuda()