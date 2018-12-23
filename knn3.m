function knn3
   NUM1tr= xlsread('hand_writing_digits_train_1',1,'B2:BN565'); %  training set 10% of data
   X1= NUM1tr(1:564, 1:64);
   Y1= NUM1tr(1:564, 65);      
   
   obj1 = fitcknn(X1,Y1,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl1 = crossval(obj1);
   cvmdlloss1 = kfoldLoss(cvmdl1);  %10 fold validation
   crossvalerrorrate1= cvmdlloss1
   
   NUM1t= xlsread('hand_writing_digits_test_1',1,'B2:BN5057');
   test1= NUM1t(1:5056, 1:64);
   label1 = predict(obj1,test1);
   actual1 = NUM1t(1:5056, 65);  
   error=0;
   
   for i=1:5056
     if (label1(i)- actual1(i)~=0)
         error=error+1;
     end
   end
     
      errorrate1= error/5056
   
     %  training set 20% of data 
   NUM2tr= xlsread('hand_writing_digits_train_2',1,'B2:BN1127'); 
   X2= NUM2tr(1:1126, 1:64);
   Y2= NUM2tr(1:1126, 65);      
   
   obj2 = fitcknn(X2,Y2,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl2 = crossval(obj2);
   cvmdlloss2 = kfoldLoss(cvmdl2);  %10 fold validation
   crossvalerrorrate2= cvmdlloss2
   
   NUM2t= xlsread('hand_writing_digits_test_2',1,'B2:BN4495');
   test2= NUM2t(1:4494, 1:64);
   label2 = predict(obj2,test2);
   actual2 = NUM2t(1:4494, 65);  
   error=0;
   
   for i=1:4494
     if (label2(i)- actual2(i)~=0)
         error=error+1;
     end
   end
     
      errorrate2= error/4494   
      
   %  training set 30% of data 
   NUM3tr= xlsread('hand_writing_digits_train_3',1,'B2:BN1688'); 
   X3= NUM3tr(1:1687, 1:64);
   Y3= NUM3tr(1:1687, 65);      
   
   obj3 = fitcknn(X3,Y3,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl3 = crossval(obj3);
   cvmdlloss3 = kfoldLoss(cvmdl3);  %10 fold validation
   crossvalerrorrate3= cvmdlloss3
   
   NUM3t= xlsread('hand_writing_digits_test_3',1,'B2:BN3934');
   test3= NUM3t(1:3933, 1:64);
   label3 = predict(obj3,test3);
   actual3 = NUM3t(1:3933, 65);  
   error=0;
   
   for i=1:3933
     if (label3(i)- actual3(i)~=0)
         error=error+1;
     end
   end
     
      errorrate3= error/3933   
          
      %  training set 40% of data 
   NUM4tr= xlsread('hand_writing_digits_train_4',1,'B2:BN2250'); 
   X4= NUM4tr(1:2249, 1:64);
   Y4= NUM4tr(1:2249, 65);      
   
   obj4 = fitcknn(X4,Y4,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl4 = crossval(obj4);
   cvmdlloss4 = kfoldLoss(cvmdl4);  %10 fold validation
   crossvalerrorrate4= cvmdlloss4
   
   NUM4t= xlsread('hand_writing_digits_test_4',1,'B2:BN3372');
   test4= NUM4t(1:3371, 1:64);
   label4 = predict(obj4,test4);
   actual4 = NUM4t(1:3371, 65);  
   error=0;
   
   for i=1:3371
     if (label4(i)- actual4(i)~=0)
         error=error+1;
     end
   end
     
      errorrate4= error/3371
      
    %  training set 50% of data 
   NUM5tr= xlsread('hand_writing_digits_train_5',1,'B2:BN2811'); 
   X5= NUM5tr(1:2810, 1:64);
   Y5= NUM5tr(1:2810, 65);      
   
   obj5 = fitcknn(X5,Y5,'NumNeighbors',5,'Standardize',1);  % knn classifier with k=5
   cvmdl5 = crossval(obj5);
   cvmdlloss5 = kfoldLoss(cvmdl5);  %10 fold validation
   crossvalerrorrate5= cvmdlloss5
   
   NUM5t= xlsread('hand_writing_digits_test_5',1,'B2:BN2811');
   test5= NUM5t(1:2810, 1:64);
   label5 = predict(obj5,test5);
   actual5 = NUM5t(1:2810, 65);  
   error=0;
   
   for i=1:2810
     if (label5(i)- actual5(i)~=0)
         error=error+1;
     end
   end
     
      errorrate5= error/2810