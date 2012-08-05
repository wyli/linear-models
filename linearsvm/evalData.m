samples = [0.814723686393179,0.126986816293506,0.632359246225410;0.905791937075619,0.913375856139019,0.097540404999410];
targets = [1;1;-1];
plot(samples(1,1), samples(2,1), 'r+');
hold on;
plot(samples(1,2), samples(2,2), 'r+');
hold on;
plot(samples(1,3), samples(2, 3), 'g+');
smo = smosvm(samples, targets);
y1 = evalSvm(smo, samples(:,1), samples, targets);
y2 = evalSvm(smo, samples(:,2), samples, targets);
y3 = evalSvm(smo, samples(:,3), samples, targets);
smo.Error
