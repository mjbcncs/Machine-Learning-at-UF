function knn1
   [data1tr,text1tr]= xlsread('breast_cancer_train_1',1,'B2:K70'); %  training set 10% of data
   X1= data1tr(1:69, 1:9);
   Y1= text1tr(1:69, 1);      
   
   obj1 = fitcknn(X1,Y1,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl1 = crossval(obj1);
   cvmdlloss1 = kfoldLoss(cvmdl1);  %10 fold validation
   crossvalerrorrate1= cvmdlloss1
   
   [data1t,text1t]= xlsread('breast_cancer_test_1',1,'B2:K615');
   test1= data1t(1:614, 1:9);
   label1 = predict(obj1,test1);
   actual1 = text1t(1:614, 1);  
   error=0;
   
   for i=1:614
     if (strcmp(label1(i),actual1(i))~=1)
         error=error+1;
     end
   end
     
      errorrate1= error/614
         
   %  training set 20% of data  
   [data2tr,text2tr]= xlsread('breast_cancer_train_2',1,'B2:K138');
   X2= data2tr(1:137, 1:9);
   Y2= text2tr(1:137, 1);      
   
   obj2 = fitcknn(X2,Y2,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl2 = crossval(obj2);
   cvmdlloss2 = kfoldLoss(cvmdl2);  %10 fold validation
   crossvalerrorrate2= cvmdlloss2
   
   [data2t,text2t]= xlsread('breast_cancer_test_2',1,'B2:K547');
   test2= data2t(1:546, 1:9);
   label2 = predict(obj2,test2);
   actual2 = text2t(1:546, 1);  
   error=0;
   
   for i=1:546
     if (strcmp(label2(i),actual2(i))~=1)
         error=error+1;
     end
   end
     
      errorrate2= error/546      
      
      %  training set 30% of data  
   [data3tr,text3tr]= xlsread('breast_cancer_train_3',1,'B2:K207');
   X3= data3tr(1:206, 1:9);
   Y3= text3tr(1:206, 1);      
   
   obj3 = fitcknn(X3,Y3,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl3 = crossval(obj3);
   cvmdlloss3 = kfoldLoss(cvmdl3);  %10 fold validation
   crossvalerrorrate3= cvmdlloss3
   
   [data3t,text3t]= xlsread('breast_cancer_test_3',1,'B2:K478');
   test3= data3t(1:477, 1:9);
   label3 = predict(obj3,test3);
   actual3 = text3t(1:477, 1);  
   error=0;
   
   for i=1:477
     if (strcmp(label3(i),actual3(i))~=1)
         error=error+1;
     end
   end
     
      errorrate3= error/477     
      
  %  training set 40% of data  
   [data4tr,text4tr]= xlsread('breast_cancer_train_4',1,'B2:K275');
   X4= data4tr(1:274, 1:9);
   Y4= text4tr(1:274, 1);      
   
   obj4 = fitcknn(X4,Y4,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl4 = crossval(obj4);
   cvmdlloss4 = kfoldLoss(cvmdl4);  %10 fold validation
   crossvalerrorrate4= cvmdlloss4
   
   [data4t,text4t]= xlsread('breast_cancer_test_4',1,'B2:K410');
   test4= data4t(1:409, 1:9);
   label4 = predict(obj4,test4);
   actual4 = text4t(1:409, 1);  
   error=0;
   
   for i=1:409
     if (strcmp(label4(i),actual4(i))~=1)
         error=error+1;
     end
   end
     
      errorrate4= error/409    
      
      
    %  training set 50% of data  
   [data5tr,text5tr]= xlsread('breast_cancer_train_5',1,'B2:K343');
   X5= data5tr(1:342, 1:9);
   Y5= text5tr(1:342, 1);      
   
   obj5 = fitcknn(X5,Y5,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl5 = crossval(obj5);
   cvmdlloss5 = kfoldLoss(cvmdl5);  %10 fold validation
   crossvalerrorrate5= cvmdlloss5
   
   [data5t,text5t]= xlsread('breast_cancer_test_5',1,'B2:K342');
   test5= data5t(1:341, 1:9);
   label5 = predict(obj5,test5);
   actual5 = text5t(1:341, 1);  
   error=0;
   
   for i=1:341
     if (strcmp(label5(i),actual5(i))~=1)
         error=error+1;
     end
   end
     
      errorrate5= error/341
      
      