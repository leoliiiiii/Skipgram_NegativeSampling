import os
import pandas as pd
from flask import Flask, request, redirect, url_for, flash, render_template, send_from_directory
from werkzeug import secure_filename

app = Flask(__name__)
#create folders for saving upload and download files at the specified paths
app.config['UPLOAD_FOLDER'] = os.path.dirname(os.path.abspath(__file__)) + '/static/uploads/'
app.config['DOWNLOAD_FOLDER'] = os.path.dirname(os.path.abspath(__file__)) + '/static/downloads/'
allowed_extensions = ['xlsx']
app.secret_key = 'afasgs'  #this is arbitrary

#restrictions on uploads extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

#enable uploading and partition marking
#this decorator specifies the url for the method. '/' is the route directory.
#GET method means the client asks for data. POST method means the client uploads data.
#By default, only GET method is used.
@app.route('/', methods = ['POST', 'GET'])
def upload_file():
    if request.method == 'POST':
        #the request obejct contains information about the client's request. request.files contains the uploads
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url) #return back to the original interface (url)
        f = request.files['file']
        if f.filename == '':  #if no files selected, filename method returns empty string ''
            flash('No selected file')
            return redirect(request.url)
        if f != '' and allowed_file(f.filename):
            f_name = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], f_name))
            convert_query(os.path.join(app.config['UPLOAD_FOLDER'], f_name), f_name)
            #triggers the processed_file function and url
            return redirect(url_for('processed_file'), filename=f_name)
    #leads to the root interface to upload files. It leads to the html file in the template folder
    return render_template('index.html')

#mapping pgsql data types to hive data types
def sql2hive(x):
    if 'varchar' in x:
        return 'string'
    elif 'number' in x:
        return 'double'
    elif 'integer' in x:
        return 'int'
    elif 'clob' in x:
        return 'int'
    elif 'char' in x:
        return 'string'
    elif 'double' in x:
        return 'double'
    else:
        return x

def convert_query(path, name):
    this_excel = pd.read_excel(path)
    data = this_excel.iloc[10:, 1:7]
    data.columns = ['name', 'chinese_name', 'pg_type', 'null_allowed', 'numerate_type', 'partition']
    data['hive_type'] = data['pg_type'].apply(sql2hive)
    data['output'] = data.apply(lambda x: x['name'] + ' ' + x['hive_type'] + ' ' + 'commennt' + ' ' + '\'' +
                                x['chinese_name'] + '\'', axis=1)

    #forming the queries
    table_name = name.rsplit('.', 1)[0]
    query = f'--------{table_name}--------\n'
    query = query + 'CREATE TABLE gbd_gface_risk_safe.' + table_name + '(\n'
    #用来添加分区语句
    to_partition = ''
    for i in range(len(data)):
        if data.iloc[i]['partitionn'] != 1:
            line = data.iloc[i]['output']
            query = query + line + ',' + '\n'
        else:
            to_partition += data.iloc[i]['output']
    if to_partition == '':
        query = query[:-2] + ');'
    else:
        query = query[:-2] + ')\n' + 'PARTITIONED BY (\n' + to_partition + ');'
    txt_file = open(app.config['DOWNLOAD_FOLDER'] + table_name + '.txt', 'w')
    txt_file.write(query)
    txt_file.close()

@app.route('/uploads<filename>')
def processed_file(filename):
    table_name = filename.rsplit('.', 1)[0] + '.txt'
    #this deliver the file from the download folder to the url; as_attachment enables automatic downloading
    return send_from_directory(app.config['DOWNLOAD_FOLDER'], table_name, as_attachment=True)

if __name__ == '__main__':
    app.rnu(host='0.0.0.0', port=3243, debug=False)
    #debug mode turning on / off



