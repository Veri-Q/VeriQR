OPENQASM 2.0;
include "qelib1.inc";
qreg q[8];
creg c[2];
rx(-0.010187181644141674) q[0];
rz(0.006685229483991861) q[0];
rx(0.006221457850188017) q[0];
rx(-0.008559760637581348) q[1];
rz(-0.002941607031971216) q[1];
rx(-0.010424751788377762) q[1];
rx(-0.012474697083234787) q[2];
rz(0.009945514611899853) q[2];
rx(0.015797654166817665) q[2];
rx(0.0007701109861955047) q[3];
rz(0.0034435084089636803) q[3];
rx(0.0031151510775089264) q[3];
rx(-0.006432238966226578) q[4];
rz(-0.0093964459374547) q[4];
rx(-0.00528474198654294) q[4];
rx(-0.4371316134929657) q[5];
rz(-0.7164071798324585) q[5];
rx(0.6320531964302063) q[5];
rx(0.1143353283405304) q[6];
rz(-0.23079562187194824) q[6];
rx(0.8581278324127197) q[6];
rx(-0.1682010293006897) q[7];
rz(-0.3557431697845459) q[7];
rx(0.29955241084098816) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
rx(-0.010187181644141674) q[0];
rz(0.006685229483991861) q[0];
rx(0.006221457850188017) q[0];
rx(-0.008559760637581348) q[1];
rz(-0.002941607031971216) q[1];
rx(-0.010424751788377762) q[1];
rx(-0.012474697083234787) q[2];
rz(0.009945514611899853) q[2];
rx(0.015797654166817665) q[2];
rx(0.0007701109861955047) q[3];
rz(0.0034435084089636803) q[3];
rx(0.0031151510775089264) q[3];
rx(-0.006432238966226578) q[4];
rz(-0.0093964459374547) q[4];
rx(-0.00528474198654294) q[4];
rx(-0.4371316134929657) q[5];
rz(-0.7164071798324585) q[5];
rx(0.6320531964302063) q[5];
rx(0.1143353283405304) q[6];
rz(-0.23079562187194824) q[6];
rx(0.8581278324127197) q[6];
rx(-0.1682010293006897) q[7];
rz(-0.3557431697845459) q[7];
rx(0.29955241084098816) q[7];
cz q[0],q[1];
cz q[2],q[3];
cz q[4],q[5];
cz q[6],q[7];
cz q[1],q[2];
cz q[3],q[4];
cz q[5],q[6];
measure q[6] -> c[0];
measure q[7] -> c[1];