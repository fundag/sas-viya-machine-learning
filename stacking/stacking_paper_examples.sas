/* Start a CAS session named mySession */
cas mySession; 

/* Define a CAS engine libref for CAS in-memory data tables */
libname cas sasioca; 

/* Create a SAS libref for the directory that has the data */
libname data "/folders/myfolders/"; 

/* Load level-2 OOF predictions into CAS using a DATA step */
data cas.train_mean_oofs;
	set data.train_oofs;
run;

/* Load level-2 test set predictions into CAS using a DATA step */
data cas.test_mean_preds;
	set data.test_preds;
run;

/************************/
/* Asses level 2 models */
/************************/

/* Calculate cross validation errors */
data cas.ase_train;
   set cas.train_mean_oofs;
   se_gbt=(mean_gbt-target)*(mean_gbt-target);
   se_frst=(mean_frst-target)*(mean_frst-target);
   se_logit=(mean_logit-target)*(mean_logit-target);
   se_factmac=(mean_factmac-target)*(mean_factmac-target);
run;

/* Calculate test errors */
data cas.ase_test;
   set cas.test_mean_preds;
   se_gbt=(mean_gbt-target)*(mean_gbt-target);
   se_frst=(mean_frst-target)*(mean_frst-target);
   se_logit=(mean_logit-target)*(mean_logit-target);
   se_factmac=(mean_factmac-target)*(mean_factmac-target);
run;

proc cas;
   summary/ table={name='ase_train', vars={'se_gbt', 'se_frst', 'se_logit','se_factmac'}}; 
   summary/ table={name='ase_test', vars={'se_gbt', 'se_frst', 'se_logit','se_factmac'}}; 
run;
quit;


title "Stacking with adaptive lasso";
/* Train a regression model with adaptive lasso */
proc regselect data=cas.train_mean_oofs;
   partition fraction(validate=0.3);
   model target=mean_factmac mean_gbt mean_logit mean_frst /noint;
               selection method=lasso(adaptive stop=sbc choose=validate) details=all;
   code file="/u/fgunes/cas/adult/ensemble/lasso_score.sas";
run;

/* Score test set */
data cas.lasso_score;
   set CAS.test_mean_preds;
   %include '/u/fgunes/cas/adult/ensemble/lasso_score.sas';
run;

/* Calculate test error */
data cas.lasso_score;
   set cas.lasso_score;
   se=(p_target-target)*(p_target-target);
run;

proc cas;
   summary/ table={name='lasso_score', vars={'se'}}; 
run;
quit;


title "Stacking with nonnegative least squares";
/* Train a nonnegative least squares regression model */
proc cqlim data=cas.train_mean_oofs;
   model target= mean_gbt mean_factmac mean_frst mean_logit  / noint;
   restrict mean_gbt>0;
   restrict mean_frst>0;
   restrict mean_logit>0;
   restrict mean_factmac>0;
   output out=cas.cqlim_train_preds xbeta copyvar=target;
run;

/* Score test set and calculate test error */
data cas.cqlim_test_preds(keep=p_target target ase);
   set cas.test_mean_preds;
   p_target=0.92143*mean_gbt+
            0.02234*mean_factmac+
            0.056071*mean_frst
            ;
   se=(p_target-target)*(p_target-target);
run;

proc cas; 
   summary/ table={name='cqlim_test_preds', vars={'se'}}; 
run;
quit;


title "Stacking with gradient boosting";
/**********************************************************************************/
/* Trains for the optimal gradient boosting model with autotuning                 */
/* Expect a long run time due to autonuning options: samplesize=200 and kfold=5   */
/* which requires fitting and scoring for 1000 models                             */
/* For shorter run time, decrease the samplesize                                  */
/**********************************************************************************/
proc gradboost data=cas.train_mean_oofs outmodel=cas.gbt_ensemble;
   target target / level=nominal;
   input mean_factmac mean_gbt mean_logit mean_frst / level=interval;
   autotune tuningparameters=(ntrees samplingrate vars_to_try(init=4) learningrate(ub=0.3) lasso ridge) 
            searchmethod=random samplesize=200 objective=ase kfold=5;
   ods output FitStatistics=Work._Gradboost_FitStats_ 
              VariableImportance=Work._Gradboost_VarImp_;
run;

/* Score test set */
proc gradboost  data=cas.test_mean_preds inmodel=cas.gbt_ensemble;   
   output out=cas.test_gbtscr copyvars=(id target);
run;

/* Calculate test error */
data cas.test_gbtscr;
   set cas.test_gbtscr;
   se=(p_target1-target)*(p_target1-target);
run;

proc cas;
   summary/ table={name='test_gbtscr', vars={'se'}}; 
run;
quit;
