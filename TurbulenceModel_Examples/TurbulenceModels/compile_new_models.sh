cd ../../
source prep_env.sh

cd TurbulenceModel_Examples/TurbulenceModels/momentumTransportModels
# wclean
wmake

cd ../incompressible
# wclean
wmake

cd ../