// The help of each Digital Twin screen and analysis, in both languages. It is
// added to the ROSS interface's help (`addHelp` in /components/help.js) under
// `dt-<key>` and opened with its `openSectionHelp`, in its modal.

const HELP = {
    dt: {
        pt: {
            title: 'Digital Twin',
            body: `
                <h4>1. Equipamento</h4>
                <ul>
                    <li>Escolha a <b>plataforma</b> e o <b>equipamento</b>. Um equipamento marcado como indisponível usa um modelo que importa um pacote não instalado (por exemplo, <code>rossxl</code> ou <code>rossml</code>); a nota abaixo da lista diz qual.</li>
                    <li>O <b>sentido de rotação</b> é visto do acionador para o acionado.</li>
                </ul>
                <h4>2. Dados de Processo</h4>
                <ul>
                    <li>Escolha a data, a hora inicial e a <b>origem dos dados</b>, e clique em <b>Carregar dados</b>. A tabela aparece no painel à direita.</li>
                    <li><b>PI Data</b> lê a janela de 15 minutos no servidor PI; precisa do pacote <code>pandaspi</code> e das variáveis de ambiente <code>USER_SWAGGER</code> e <code>SWAGGER_PASSWORD</code>. <b>Dados de amostra</b> lê o arquivo .parquet que acompanha o modelo.</li>
                    <li>Células marcadas como <b>vazio</b> são valores nulos e podem comprometer o modelo.</li>
                </ul>
                <h4>3. Modelo</h4>
                <ul>
                    <li>As <b>configurações adicionais</b> (folga radial dos mancais, temperatura do óleo, ...) vêm preenchidas com os valores do modelo.</li>
                    <li><b>Criar modelo</b> monta o rotor a partir dos dados de processo e salva o arquivo <b>.toml</b> na sessão.</li>
                    <li><b>Abrir no Rotor Hub</b> adiciona o rotor à biblioteca da interface, onde ele pode ser editado e analisado como qualquer outro.</li>
                    <li>A primeira criação demora alguns segundos enquanto o ROSS é importado.</li>
                </ul>
                <h4>4. Balanceamento</h4>
                <ul>
                    <li>Os <b>planos de medição</b> são os nós onde há sensores. Cada um recebe dois sensores, X e Y, e o sensor Y fica 90° depois do X, então só o ângulo do X é informado. São necessários pelo menos dois planos.</li>
                    <li>Um <b>plano de balanceamento</b> é um nó e o <b>raio</b>, em milímetros, onde a massa de correção será colocada: a estimativa é massa × raio, e o raio a transforma em massa.</li>
                    <li>No cartão <b>Sensores</b>, envie o sinal de cada sensor em <b>.csv</b> e clique em <b>Aplicar</b>. O arquivo deve ser o exportado pelo sistema de medição, com o cabeçalho que traz a velocidade e as unidades: é de lá que a análise lê a rotação dos sinais.</li>
                    <li><b>Executar análise</b> estima o desbalanceamento com o filtro de Kalman do <code>rossml</code> e mostra, por plano, a massa de correção e a fase. A fase da correção é a do desbalanceamento mais 180°.</li>
                    <li>Os <b>parâmetros da análise</b> são escolhidos numa lista: os padrões do <code>rossml</code> ou os parâmetros de uma máquina do app do Digital Twin, quando ele está neste computador. Um modelo pré-estabelecido começa pelos da sua máquina. Em <b>Ajustar valores</b>, cada valor (sinal, filtro de Kalman, limites da força resultante, fatores do filtro e seus limites de busca) pode ser mudado; um campo vazio usa o padrão do <code>rossml</code>.</li>
                    <li><b>Iterações extras</b> e <b>população</b> controlam o otimizador; as iterações padrão são 24 × (1 + extras). A <b>otimização de planos</b> escolhe o subconjunto de planos que melhor explica os sinais e triplica o tempo.</li>
                    <li>O cartão <b>Resultado do balanceamento</b> traz a tabela por plano e o <b>gráfico polar</b>: azul é o desbalanceamento estimado, verde é a correção. O gráfico é orientado pelo ângulo de um sensor, então escolha o <b>sensor de referência</b>. A fase da tabela acompanha o sensor escolhido; a massa não depende dele.</li>
                    <li>Em <b>Gráfico</b> escolha o que ver da mesma execução: a <b>correção decomposta</b> em duas posições (ângulo e raio de cada uma, depois <b>Decompor</b>), as <b>forças estimadas</b> em cada plano de balanceamento (tempo e órbita) e o <b>sensoriamento virtual</b>, os deslocamentos que o filtro reconstrói nos planos de balanceamento e de medição (tempo e órbita) e ao longo do rotor (3D).</li>
                    <li>A análise precisa do pacote <code>rossml</code> instalado. Sem ele, o botão fica desabilitado e o painel diz o que falta.</li>
                </ul>`,
        },
        en: {
            title: 'Digital Twin',
            body: `
                <h4>1. Equipment</h4>
                <ul>
                    <li>Choose the <b>platform</b> and the <b>equipment</b>. Equipment marked unavailable uses a model that imports a package that is not installed (for example <code>rossxl</code> or <code>rossml</code>); the note below the list says which.</li>
                    <li>The <b>rotation direction</b> is seen from the driver towards the driven machine.</li>
                </ul>
                <h4>2. Process Data</h4>
                <ul>
                    <li>Choose the date, the start time and the <b>data source</b>, and click <b>Load data</b>. The table appears in the right panel.</li>
                    <li><b>PI Data</b> reads the 15-minute window from the PI server; it needs the <code>pandaspi</code> package and the <code>USER_SWAGGER</code> and <code>SWAGGER_PASSWORD</code> environment variables. <b>Sample data</b> reads the .parquet file kept with the model.</li>
                    <li>Cells marked <b>empty</b> are null values and may compromise the model.</li>
                </ul>
                <h4>3. Model</h4>
                <ul>
                    <li>The <b>additional settings</b> (bearing radial clearance, oil temperature, ...) start with the model's values.</li>
                    <li><b>Create model</b> builds the rotor from the process data and saves the <b>.toml</b> file in the session.</li>
                    <li><b>Open in Rotor Hub</b> adds the rotor to the interface's library, where it is edited and analysed like any other.</li>
                    <li>The first creation takes a few seconds while ROSS is imported.</li>
                </ul>
                <h4>4. Balancing</h4>
                <ul>
                    <li>The <b>measuring planes</b> are the nodes with sensors. Each one takes two sensors, X and Y, and the Y sensor sits 90° after the X sensor, so only the X angle is given. At least two planes are needed.</li>
                    <li>A <b>balancing plane</b> is a node and the <b>radius</b>, in millimetres, where the correction mass goes: the estimate is a mass times a radius, and the radius turns it into a mass.</li>
                    <li>In the <b>Sensors</b> card, send each sensor's signal as a <b>.csv</b> and click <b>Apply</b>. The file must be the one the measuring system exports, with the header that carries the speed and the units: that is where the analysis reads the speed of the signals from.</li>
                    <li><b>Run analysis</b> estimates the unbalance with the Kalman filter of <code>rossml</code> and shows, per plane, the correction mass and its phase. The correction's phase is the unbalance's plus 180°.</li>
                    <li>The <b>analysis parameters</b> are chosen from a list: the <code>rossml</code> defaults, or the parameters of a machine of the Digital Twin app when it is on this computer. A pre-established model starts from its machine's. Under <b>Adjust values</b>, each value (signal, Kalman filter, resultant force limits, filter factors and their search bounds) can be changed; an empty field uses the <code>rossml</code> default.</li>
                    <li><b>Extra iterations</b> and <b>population</b> drive the optimizer; its own iterations are 24 × (1 + extra). The <b>plane optimization</b> picks the subset of planes that explains the signals best and triples the running time.</li>
                    <li>The <b>Balancing result</b> card carries the table per plane and the <b>polar chart</b>: blue is the estimated unbalance, green the correction. The chart is turned to one sensor's angle, so choose the <b>reference sensor</b>. The phase in the table follows the sensor chosen; the mass does not depend on it.</li>
                    <li>In <b>Chart</b>, choose what to see from the same run: the <b>correction split</b> into two positions (the angle and radius of each, then <b>Split</b>), the <b>estimated forces</b> on each balancing plane (time and orbit) and the <b>virtual sensing</b>, the displacements the filter reconstructs on the balancing and measuring planes (time and orbit) and along the rotor (3D).</li>
                    <li>The analysis needs the <code>rossml</code> package installed. Without it the button stays off and the panel says what is missing.</li>
                </ul>`,
        },
    },
    balance: {
        pt: {
            title: 'BRBalance',
            body: `
                <h4>Configuração</h4>
                <ul><li>Defina quantos <b>sensores</b>, <b>testes</b>, <b>planos</b> e <b>massas por plano</b> o procedimento usa. As tabelas das outras abas acompanham esses números.</li></ul>
                <h4>Referência, Rodadas de Teste e Condição Atual</h4>
                <ul>
                    <li>Amplitude e fase de cada sensor. A fase é em graus.</li>
                    <li>Em cada teste, informe as massas adicionadas ou removidas em cada plano; a linha <b>Total</b> soma as massas vetorialmente.</li>
                </ul>
                <h4>Correções Aplicadas</h4>
                <ul><li>Entram só na vibração prevista; não alteram a correção recomendada.</li></ul>
                <h4>Arquivos</h4>
                <ul>
                    <li><b>Abrir</b> e <b>Salvar</b> leem e gravam o formulário como arquivo <b>.brb</b>.</li>
                    <li><b>Relatório</b> baixa o relatório Word do cálculo.</li>
                </ul>`,
        },
        en: {
            title: 'BRBalance',
            body: `
                <h4>Setup</h4>
                <ul><li>Set how many <b>sensors</b>, <b>tests</b>, <b>planes</b> and <b>weights per plane</b> the procedure uses. The tables of the other tabs follow these numbers.</li></ul>
                <h4>Reference, Test Runs and Current Condition</h4>
                <ul>
                    <li>Amplitude and phase of each sensor. The phase is in degrees.</li>
                    <li>In each test, enter the weights added or removed on each plane; the <b>Total</b> row adds them as vectors.</li>
                </ul>
                <h4>Applied Corrections</h4>
                <ul><li>They only enter the predicted vibration; they do not change the recommended correction.</li></ul>
                <h4>Files</h4>
                <ul>
                    <li><b>Open</b> and <b>Save</b> read and write the form as a <b>.brb</b> file.</li>
                    <li><b>Report</b> downloads the Word report of the calculation.</li>
                </ul>`,
        },
    },

    analyses: {
        pt: { title: 'Análises', body: `
            <h4>Painéis</h4>
            <ul>
                <li>Escolha a análise no painel escuro e clique em <b>Adicionar Painel</b>. O painel calcula com os valores padrão assim que aparece.</li>
                <li>Altere os campos abaixo do gráfico e clique em <b>Atualizar</b> para recalcular.</li>
                <li>Todas as análises usam o modelo criado na etapa 3. Criar o modelo de novo apaga os painéis, porque eles passariam a descrever outro rotor.</li>
            </ul>` },
        en: { title: 'Analyses', body: `
            <h4>Dashboards</h4>
            <ul>
                <li>Pick the analysis in the dark panel and click <b>Add Dashboard Card</b>. The card computes with the default values as soon as it appears.</li>
                <li>Change the fields under the chart and click <b>Update</b> to compute again.</li>
                <li>Every analysis uses the model created in step 3. Creating the model again removes the cards, since they would describe another rotor.</li>
            </ul>` },
    },
    campbell: {
        pt: { title: 'Diagrama de Campbell', body: `
            <ul>
                <li>Frequências naturais amortecidas em função da velocidade de rotação, entre a velocidade inicial e a final.</li>
                <li><b>Harmônicos</b>: linhas de excitação 1x, 2x, ... separadas por vírgula (1 a 15).</li>
                <li>Clique num ponto do diagrama para ver, ao lado, o modo de forma 3D daquela velocidade e frequência.</li>
            </ul>` },
        en: { title: 'Campbell Diagram', body: `
            <ul>
                <li>Damped natural frequencies against rotation speed, between the initial and the final speed.</li>
                <li><b>Harmonics</b>: excitation lines 1x, 2x, ... separated by commas (1 to 15).</li>
                <li>Click a point on the diagram to see the 3D mode shape at that speed and frequency beside it.</li>
            </ul>` },
    },
    unbalance: {
        pt: { title: 'Resposta ao Desbalanceamento', body: `
            <ul>
                <li>O desbalanceamento residual da API 684, <b>Umax = 6350·W/N</b> [g·mm] (W: massa do rotor em kg; N: velocidade máxima contínua em RPM, limitada a 25000), multiplicado pelo percentual informado, é distribuído entre os discos em proporção à massa de cada um, com fases aleatórias.</li>
                <li><b>Desbalanceamentos adicionais</b> entram em g·mm, no nó e na fase informados.</li>
                <li>À esquerda, o diagrama de Bode nos dois probes (amplitude em µm pico a pico). À direita, a deformada 3D na velocidade máxima contínua.</li>
                <li>A tabela abaixo lista todos os desbalanceamentos aplicados, inclusive as fases sorteadas.</li>
            </ul>` },
        en: { title: 'Unbalance Response', body: `
            <ul>
                <li>The API 684 residual unbalance, <b>Umax = 6350·W/N</b> [g·mm] (W: rotor mass in kg; N: maximum continuous speed in RPM, capped at 25000), times the given percentage, is spread over the disks in proportion to their mass, at random phases.</li>
                <li><b>Additional unbalances</b> are entered in g·mm, at the given node and phase.</li>
                <li>On the left, the Bode plot at both probes (amplitude in µm peak to peak). On the right, the 3D deflected shape at the maximum continuous speed.</li>
                <li>The table below lists every unbalance applied, drawn phases included.</li>
            </ul>` },
    },
    clearance: {
        pt: { title: 'Análise de Folgas', body: `
            <ul>
                <li>O desbalanceamento residual da API na velocidade informada é aplicado no nó de maior amplitude do primeiro modo com precessão direta (Forward).</li>
                <li>A resposta nos mancais é escalada para que a maior amplitude seja <b>A1 = 25,4·√(12000/N)</b> µm pico a pico.</li>
                <li>As barras são a folga radial de cada mancal (vermelho) e 75% dela (azul). O texto sobre a curva é a amplitude como % de 75% da folga / % da folga.</li>
            </ul>` },
        en: { title: 'Clearance Analysis', body: `
            <ul>
                <li>The API residual unbalance at the given speed is applied at the node of largest amplitude of the first forward mode.</li>
                <li>The response at the bearings is scaled so the largest amplitude equals <b>A1 = 25.4·√(12000/N)</b> µm peak to peak.</li>
                <li>The bars are each bearing's radial clearance (red) and 75% of it (blue). The text on the curve is the amplitude as % of 75% of the clearance / % of the clearance.</li>
            </ul>` },
    },
    stability: {
        pt: { title: 'Análise de Estabilidade', body: `
            <ul>
                <li>Direção de precessão, frequência amortecida e decremento logarítmico de cada modo, na velocidade informada, com o modelo de 4 graus de liberdade por nó.</li>
                <li>Duas tabelas: só rotor e mancais, e rotor com mancais e selos. A diferença entre as duas mostra o efeito dos selos na estabilidade.</li>
            </ul>` },
        en: { title: 'Stability Analysis', body: `
            <ul>
                <li>Whirl direction, damped frequency and logarithmic decrement of each mode at the given speed, with the 4 degrees of freedom per node model.</li>
                <li>Two tables: rotor and bearings only, and rotor with bearings and seals. The difference between them shows what the seals do to the stability.</li>
            </ul>` },
    },
};

const ICON = "<i class='fas fa-question-circle'></i> ";

export const HELP_ENTRIES = Object.fromEntries(Object.entries(HELP).map(([key, text]) => [
    'dt-' + key,
    {
        title: { en: ICON + text.en.title, pt: ICON + text.pt.title },
        body: { en: text.en.body, pt: text.pt.body },
    },
]));
