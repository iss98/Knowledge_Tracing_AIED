import os
def versionFind(versionName):
    reportsList = os.listdir(os.path.join(os.getcwd(), 'reports'))
    
    possibleVersions = []
    for report in reportsList:
        if versionName.lower() in report.lower():
            possibleVersions.append(report)
    
    if len(possibleVersions)==1:
        print("version Name:", possibleVersions[0])
        return possibleVersions[0]
    
    possibleVersions = []
    versionList = versionName.split('_')
    for report in reportsList:
        for v in versionList:
            if v not in report:
                satisfied = 0
                break
            else:
                satisfied = 1
            
        if satisfied == 1:
            possibleVersions.append(report)
    
    if len(possibleVersions)==1:
        print("version Name:", possibleVersions[0])
        return possibleVersions[0]
    
    versionList = versionName.split()
    for report in reportsList:
        for v in versionList:
            if v not in report:
                satisfied = 0
                break
            else:
                satisfied = 1
            
        if satisfied == 1:
            possibleVersions.append(report)
    
    if len(possibleVersions)==1:
        print("version Name:", possibleVersions[0])
        return possibleVersions[0]
    
    print("There are no possible versions. Please write down the version name exactly.")
    print("You can find a list of version at ./reports")
    