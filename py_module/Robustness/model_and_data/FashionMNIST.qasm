OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[2];
rx(0.009864898398518562) q[0];
rz(-0.021740233525633812) q[0];
rx(-0.029092060402035713) q[0];
rx(-0.0014685823116451502) q[1];
rz(0.0005451797042042017) q[1];
rx(0.011953488923609257) q[1];
rx(-0.0020502274855971336) q[2];
rz(0.009474558755755424) q[2];
rx(0.026484597474336624) q[2];
rx(-0.004744878504425287) q[3];
rz(0.00392657145857811) q[3];
rx(0.0073982663452625275) q[3];
rx(-0.004544825758785009) q[4];
rz(0.017772389575839043) q[4];
rx(-0.0032428691629320383) q[4];
rx(-0.3187904953956604) q[5];
rz(-1.105085849761963) q[5];
rx(0.6305007934570312) q[5];
rx(0.7464872598648071) q[6];
rz(0.015428973361849785) q[6];
rx(0.8063831925392151) q[6];
rx(0.07384845614433289) q[7];
rz(-1.4472664594650269) q[7];
rx(-0.3610776662826538) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
rx(0.009864898398518562) q[0];
rz(-0.021740233525633812) q[0];
rx(-0.029092060402035713) q[0];
rx(-0.0014685823116451502) q[1];
rz(0.0005451797042042017) q[1];
rx(0.011953488923609257) q[1];
rx(-0.0020502274855971336) q[2];
rz(0.009474558755755424) q[2];
rx(0.026484597474336624) q[2];
rx(-0.004744878504425287) q[3];
rz(0.00392657145857811) q[3];
rx(0.0073982663452625275) q[3];
rx(-0.004544825758785009) q[4];
rz(0.017772389575839043) q[4];
rx(-0.0032428691629320383) q[4];
rx(-0.3187904953956604) q[5];
rz(-1.105085849761963) q[5];
rx(0.6305007934570312) q[5];
rx(0.7464872598648071) q[6];
rz(0.015428973361849785) q[6];
rx(0.8063831925392151) q[6];
rx(0.07384845614433289) q[7];
rz(-1.4472664594650269) q[7];
rx(-0.3610776662826538) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
measure q[6] -> c[0];
measure q[7] -> c[1];