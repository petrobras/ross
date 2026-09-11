import { currentLanguage, t } from '../core/i18n.js';
// The help content and the modal that shows it.

// The help content, in both languages.
//
// Until the i18n slice this was ~1240 words in English only -- the largest block
// of text in the interface, and the one that most needs to be read in the
// reader's own language. The English here is the same as before: it was
// extracted by evaluating the module, not retyped.
const HelpContent = {
    general: {
        title: {
            en: `<i class='fas fa-book'></i> Interface Guide`,
            pt: `<i class='fas fa-book'></i> Guia da Interface`,
        },
        body: {
            en: `
            <h4>Sidebar Navigation</h4>
            <p>Use the buttons on the left sidebar to switch between different modeling categories (Materials, Shafts, Disks, etc.). Use the <i class="fas fa-bars"></i> button in the topbar to hide/show the sidebar.</p>
            <h4>Adding Elements</h4>
            <p>Click the <b>+ (Add)</b> button at the bottom of the list to create a new element. You can choose different models (like BASIC, TVMS for gears, etc.) if available for that specific component.</p>
            <h4>BASIC vs LIST Creation</h4>
            <p><b>BASIC:</b> Creates a single element.<br><b>LIST:</b> Allows batch creation. Enter comma-separated values (e.g., <code>0.1, 0.2, 0.3</code>) to generate multiple elements at once. Single values will be automatically copied for all elements in the batch.</p>
            <h4>Managing the List</h4>
            <ul>
                <li><b>Confirm:</b> Click <i class="fas fa-check"></i> (green button) to save the element.</li>
                <li><b>Cancel:</b> Click <i class="fas fa-times"></i> to discard changes.</li>
                <li><b>Edit:</b> Click <i class="fas fa-pen"></i> to modify an existing element.</li>
                <li><b>Copy:</b> Click <i class="fas fa-copy"></i> to duplicate an element exactly below it.</li>
                <li><b>Delete:</b> Click <i class="fas fa-trash"></i> to remove an element permanently.</li>
            </ul>
            <h4>Export & Save Actions</h4>
            <ul>
                <li><b><i class="fab fa-python"></i> Generate Python:</b> Exports your entire rotor model and active analyses into a ready-to-run Python script.</li>
                <li><b><i class="fas fa-save"></i> Save Rotor:</b> Saves your current rotor configuration as a .json file so you can load it later without losing progress.</li>
            </ul>
        `,
            pt: `
            <h4>Navegação pelo painel lateral</h4>
            <p>Use os botões do painel à esquerda para alternar entre as categorias de modelagem (Materiais, Eixos, Discos, etc.). O botão <i class="fas fa-bars"></i> da barra superior mostra ou esconde o painel.</p>
            <h4>Acrescentar elementos</h4>
            <p>Clique em <b>+ (Adicionar)</b> no fim da lista para criar um elemento. Quando o componente tiver mais de um modelo (BASIC, TVMS para engrenagens, etc.), você escolhe qual usar.</p>
            <h4>BASIC ou LIST</h4>
            <p><b>BASIC:</b> cria um elemento.<br><b>LIST:</b> cria vários de uma vez. Informe valores separados por vírgula (ex.: <code>0.1, 0.2, 0.3</code>). Valores únicos são repetidos automaticamente para todos os elementos do lote.</p>
            <h4>Trabalhando com a lista</h4>
            <ul>
                <li><b>Confirmar:</b> <i class="fas fa-check"></i> (botão verde) salva o elemento.</li>
                <li><b>Cancelar:</b> <i class="fas fa-times"></i> descarta as alterações.</li>
                <li><b>Editar:</b> <i class="fas fa-pen"></i> altera um elemento existente.</li>
                <li><b>Copiar:</b> <i class="fas fa-copy"></i> duplica o elemento logo abaixo dele.</li>
                <li><b>Apagar:</b> <i class="fas fa-trash"></i> remove o elemento em definitivo.</li>
            </ul>
            <h4>Exportar e salvar</h4>
            <ul>
                <li><b><i class="fab fa-python"></i> Gerar Python:</b> exporta o rotor inteiro e as análises ativas como um script Python pronto para rodar.</li>
                <li><b><i class="fas fa-save"></i> Salvar Rotor:</b> guarda a configuração atual num arquivo .json, para retomar depois sem perder o trabalho.</li>
            </ul>
        `,
        },
    },
    analysis: {
        title: {
            en: `<i class='fas fa-chart-line'></i> Analysis Guide`,
            pt: `<i class='fas fa-chart-line'></i> Guia das Análises`,
        },
        body: {
            en: `
            <h4>Choosing an Analysis</h4>
            <p>Use the dropdown menu on the left sidebar to select the type of analysis you want to perform (e.g., Campbell Diagram, Unbalance Response). Click <b>⚙️ Add Dashboard Card</b> to generate it.</p>
            <h4>Using Dashboards</h4>
            <p>Each dashboard is an independent card containing a plot and its specific control parameters. <b>Manual Update System:</b> To prevent lag while setting up multiple parameters, the plot does not update automatically. Whenever you change values (like speed, node, or stiffness), you must click the <b><i class="fas fa-sync-alt"></i> Update</b> button at the top of the card to run the simulation and refresh the graph.</p>
            <h4>Managing Dashboards</h4>
            <ul>
                <li><b>Minimize/Expand:</b> Click the <i class="fas fa-chevron-down"></i> icon on the right side of the card header (or click the header itself) to hide or show the dashboard content.</li>
                <li><b>Delete:</b> Click the <i class="fas fa-trash"></i> <b>Delete</b> button to permanently remove that specific analysis card.</li>
            </ul>
            <h4>Export & Save Actions</h4>
            <ul>
                <li><b><i class="fab fa-python"></i> Generate Python:</b> Exports your rotor model and all currently active analysis cards into a Python script.</li>
                <li><b><i class="fas fa-save"></i> Save Analysis:</b> Saves all your current dashboard cards (and their parameters) as a .json file.</li>
                <li><b><i class="fas fa-folder-open"></i> Load:</b> Loads previously saved analysis dashboards from a .json file so you don't have to configure them again.</li>
            </ul>
        `,
            pt: `
            <h4>Escolher a análise</h4>
            <p>Use a lista do painel à esquerda para escolher o tipo de análise (Diagrama de Campbell, Resposta ao Desbalanceamento, etc.) e clique em <b>⚙️ Adicionar Painel</b> para criá-la.</p>
            <h4>Usando os painéis</h4>
            <p>Cada painel é um cartão independente, com um gráfico e os parâmetros daquela análise. <b>Atualização manual:</b> o gráfico não se refaz sozinho, para não recalcular a cada tecla enquanto você ajusta vários campos. Depois de mudar os valores (velocidade, nó, rigidez), clique em <b><i class="fas fa-sync-alt"></i> Atualizar</b> no topo do cartão.</p>
            <h4>Organizando os painéis</h4>
            <ul>
                <li><b>Recolher/expandir:</b> clique no <i class="fas fa-chevron-down"></i> à direita do cabeçalho (ou no próprio cabeçalho) para esconder ou mostrar o conteúdo.</li>
                <li><b>Apagar:</b> o botão <i class="fas fa-trash"></i> <b>Apagar</b> remove aquele cartão em definitivo.</li>
            </ul>
            <h4>Exportar e salvar</h4>
            <ul>
                <li><b><i class="fab fa-python"></i> Gerar Python:</b> exporta o rotor e todos os cartões ativos como um script Python.</li>
                <li><b><i class="fas fa-save"></i> Salvar Análise:</b> guarda os cartões e seus parâmetros num arquivo .json.</li>
                <li><b><i class="fas fa-folder-open"></i> Carregar:</b> traz de volta painéis salvos, sem precisar configurá-los outra vez.</li>
            </ul>
        `,
        },
    },
    materials: {
        title: {
            en: `<i class='fas fa-cube'></i> Materials Help`,
            pt: `<i class='fas fa-cube'></i> Ajuda: Materiais`,
        },
        body: {
            en: `<p>Define the physical properties of the materials used in your rotor.</p><h4>Key Parameters:</h4><ul><li><b>Density (rho):</b> Material density [kg/m³].</li><li><b>Elastic Modulus (E):</b> Young's Modulus [Pa].</li><li><b>Shear Modulus (G_s):</b> Modulus of rigidity [Pa].</li><li><b>Poisson's Ratio:</b> Transverse strain ratio.</li></ul><p><i>Note: You can assign a custom hex color for visualization in the advanced options.</i></p>`,
            pt: `<p>Define as propriedades físicas dos materiais usados no rotor.</p><h4>Parâmetros principais:</h4><ul><li><b>Densidade (rho):</b> massa específica [kg/m³].</li><li><b>Módulo de elasticidade (E):</b> módulo de Young [Pa].</li><li><b>Módulo de cisalhamento (G_s):</b> módulo de rigidez [Pa].</li><li><b>Coeficiente de Poisson:</b> razão entre deformações transversal e longitudinal.</li></ul><p><i>Nota: nas opções avançadas dá para escolher uma cor (hex) para a visualização.</i></p>`,
        },
    },
    shafts: {
        title: {
            en: `<i class='fas fa-grip-lines'></i> Shafts Help`,
            pt: `<i class='fas fa-grip-lines'></i> Ajuda: Eixos`,
        },
        body: {
            en: `<p>Shaft elements connect nodes and provide stiffness, mass, and gyroscopic effects to the rotor.</p><h4>Key Parameters:</h4><ul><li><b>Length (L):</b> Axial length of the element [mm].</li><li><b>Outer/Inner Diameters (odl, idl, odr, idr):</b> Diameters at the left (l) and right (r) sides [mm].</li><li><b>Material:</b> Link to a previously defined material name.</li></ul>`,
            pt: `<p>Os elementos de eixo ligam os nós e dão ao rotor rigidez, massa e efeitos giroscópicos.</p><h4>Parâmetros principais:</h4><ul><li><b>Comprimento (L):</b> comprimento axial do elemento [mm].</li><li><b>Diâmetros externo/interno (odl, idl, odr, idr):</b> diâmetros à esquerda (l) e à direita (r) [mm].</li><li><b>Material:</b> nome de um material já definido.</li></ul>`,
        },
    },
    disks: {
        title: {
            en: `<i class='fas fa-compact-disc'></i> Disks Help`,
            pt: `<i class='fas fa-compact-disc'></i> Ajuda: Discos`,
        },
        body: {
            en: `<p>Disks represent concentrated masses on the rotor, like impellers, couplings hubs, or turbine blades.</p><h4>Key Parameters:</h4><ul><li><b>Mass (m):</b> Disk mass [kg].</li><li><b>Polar Inertia (Ip):</b> Inertia around the rotational axis [kg.m²].</li><li><b>Diametral Inertia (Id):</b> Inertia around the transverse axis [kg.m²].</li></ul>`,
            pt: `<p>Discos representam massas concentradas no rotor: impelidores, cubos de acoplamento, pás de turbina.</p><h4>Parâmetros principais:</h4><ul><li><b>Massa (m):</b> massa do disco [kg].</li><li><b>Inércia polar (Ip):</b> inércia em torno do eixo de rotação [kg.m²].</li><li><b>Inércia diametral (Id):</b> inércia em torno do eixo transversal [kg.m²].</li></ul>`,
        },
    },
    gears: {
        title: {
            en: `<i class='fas fa-cog'></i> Gears Help`,
            pt: `<i class='fas fa-cog'></i> Ajuda: Engrenagens`,
        },
        body: {
            en: `<p>Gears act as disks but can also include meshing stiffness effects. The <b>TVMS</b> model accounts for Time-Varying Mesh Stiffness.</p><h4>Key Parameters:</h4><ul><li><b>Number of Teeth (n_teeth):</b> Gear teeth count.</li><li><b>Pitch/Base Diameter:</b> Gear sizing [mm].</li><li><b>Pressure & Helix Angles:</b> Meshing geometry [deg].</li></ul>`,
            pt: `<p>Engrenagens funcionam como discos e podem incluir o efeito da rigidez de engrenamento. O modelo <b>TVMS</b> considera a rigidez variável no tempo.</p><h4>Parâmetros principais:</h4><ul><li><b>Número de dentes (n_teeth):</b> quantidade de dentes.</li><li><b>Diâmetro primitivo/de base:</b> dimensionamento da engrenagem [mm].</li><li><b>Ângulos de pressão e de hélice:</b> geometria do engrenamento [graus].</li></ul>`,
        },
    },
    couplings: {
        title: {
            en: `<i class='fas fa-link'></i> Couplings Help`,
            pt: `<i class='fas fa-link'></i> Ajuda: Acoplamentos`,
        },
        body: {
            en: `<p>Couplings connect two different shaft sections or rotors. They add mass, inertia, and can transmit forces based on defined stiffness and damping.</p><h4>Key Parameters:</h4><ul><li><b>m_l / m_r:</b> Mass assigned to the left/right node [kg].</li><li><b>Ip_l / Ip_r:</b> Polar inertia assigned to the left/right node [kg.m²].</li><li><b>Stiffness/Damping:</b> Advanced parameters to define connection flexibility.</li></ul>`,
            pt: `<p>Acoplamentos ligam dois trechos de eixo ou dois rotores. Acrescentam massa e inércia, e transmitem forças conforme a rigidez e o amortecimento definidos.</p><h4>Parâmetros principais:</h4><ul><li><b>m_l / m_r:</b> massa atribuída ao nó esquerdo/direito [kg].</li><li><b>Ip_l / Ip_r:</b> inércia polar atribuída ao nó esquerdo/direito [kg.m²].</li><li><b>Rigidez/amortecimento:</b> parâmetros avançados que definem a flexibilidade da ligação.</li></ul>`,
        },
    },
    bearings: {
        title: {
            en: `<i class='fas fa-circle-notch'></i> Bearings Help`,
            pt: `<i class='fas fa-circle-notch'></i> Ajuda: Mancais`,
        },
        body: {
            en: `<p>Bearings support the rotor and dictate its dynamic behavior. You can use a <b>BASIC</b> spring-damper model or select specific geometries like Cylindrical, Tilting Pad, etc.</p><h4>Key Parameters (BASIC):</h4><ul><li><b>kxx, kyy, kxy, kyx:</b> Direct and cross-coupled stiffness coefficients [N/m].</li><li><b>cxx, cyy, cxy, cyx:</b> Direct and cross-coupled damping coefficients [N.s/m].</li></ul>`,
            pt: `<p>Os mancais sustentam o rotor e determinam boa parte do seu comportamento dinâmico. Use o modelo <b>BASIC</b> (mola-amortecedor) ou geometrias específicas como cilíndrico, sapatas basculantes, etc.</p><h4>Parâmetros principais (BASIC):</h4><ul><li><b>kxx, kyy, kxy, kyx:</b> coeficientes de rigidez diretos e cruzados [N/m].</li><li><b>cxx, cyy, cxy, cyx:</b> coeficientes de amortecimento diretos e cruzados [N.s/m].</li></ul>`,
        },
    },
    seals: {
        title: {
            en: `<i class='fas fa-ring'></i> Seals Help`,
            pt: `<i class='fas fa-ring'></i> Ajuda: Selos`,
        },
        body: {
            en: `<p>Seals prevent fluid leakage but introduce cross-coupled forces that can cause instability (like the Lomakin effect). Models include Labyrinth, Hole-Pattern, and Hybrid.</p><h4>Key Parameters:</h4><ul><li><b>Clearance / Radius:</b> Seal geometry [mm].</li><li><b>Inlet/Outlet Pressures:</b> Operating conditions [Pa].</li><li><b>Fluid Properties:</b> Advanced tabs usually contain gas composition and temperatures.</li></ul>`,
            pt: `<p>Selos evitam o vazamento de fluido, mas introduzem forças cruzadas que podem gerar instabilidade (como o efeito Lomakin). Há modelos labirinto, hole-pattern e híbrido.</p><h4>Parâmetros principais:</h4><ul><li><b>Folga / raio:</b> geometria do selo [mm].</li><li><b>Pressões de entrada/saída:</b> condições de operação [Pa].</li><li><b>Propriedades do fluido:</b> as abas avançadas trazem composição do gás e temperaturas.</li></ul>`,
        },
    },
    pointmasses: {
        title: {
            en: `<i class='fas fa-dot-circle'></i> Point Masses Help`,
            pt: `<i class='fas fa-dot-circle'></i> Ajuda: Massas Pontuais`,
        },
        body: {
            en: `<p>A simple concentrated mass attached to a single node, useful for modeling unbalanced weights or small components.</p><h4>Key Parameters:</h4><ul><li><b>Mass (m):</b> Total mass [kg].</li><li><b>mx, my, mz:</b> Advanced options for asymmetric mass properties.</li></ul>`,
            pt: `<p>Uma massa concentrada num único nó, útil para representar pesos de desbalanceamento ou componentes pequenos.</p><h4>Parâmetros principais:</h4><ul><li><b>Massa (m):</b> massa total [kg].</li><li><b>mx, my, mz:</b> opções avançadas para massa assimétrica.</li></ul>`,
        },
    },
    campbell: {
        title: {
            en: `<i class='fas fa-chart-line'></i> Campbell Diagram Help`,
            pt: `<i class='fas fa-chart-line'></i> Ajuda: Diagrama de Campbell`,
        },
        body: {
            en: `
            <p>A Campbell Diagram displays the system's natural frequencies as a function of the rotor's rotational speed. It is essential for tracking gyroscopic effects and predicting critical speeds.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Start/End Speed:</b> Rotational speed range boundaries for the analysis sweep [rad/s].</li>
                <li><b>Steps:</b> Number of evaluation intervals. Higher values generate smoother curves but slightly increase computation time.</li>
            </ul>
        `,
            pt: `
            <p>O diagrama de Campbell mostra as frequências naturais do sistema em função da velocidade de rotação. É essencial para acompanhar os efeitos giroscópicos e prever as velocidades críticas.</p>
            <h4>Parâmetros principais:</h4>
            <ul>
                <li><b>Velocidade inicial/final:</b> limites da faixa de rotação varrida na análise [rad/s].</li>
                <li><b>Passos:</b> número de intervalos avaliados. Mais passos dão curvas mais suaves e custam um pouco mais de tempo.</li>
            </ul>
        `,
        },
    },
    ucs: {
        title: {
            en: `<i class='fas fa-map-marked-alt'></i> UCS Diagram Help`,
            pt: `<i class='fas fa-map-marked-alt'></i> Ajuda: Diagrama UCS`,
        },
        body: {
            en: `
            <p>The Undamped Critical Speed (UCS) Map plots the rotor's natural frequencies against a broad logarithmic spectrum of support stiffness. It helps engineers identify optimal stiffness targets for bearings configuration.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Min/Max Stiffness:</b> Bearing stiffness boundaries defined exponentially (10^x N/m). For example, 4 equals 10^4 N/m and 10 equals 10^10 N/m.</li>
                <li><b>Nº Modes:</b> Total number of operational vibration modes solved and displayed on the plot grid.</li>
            </ul>
        `,
            pt: `
            <p>O mapa de velocidades críticas não amortecidas (UCS) traça as frequências naturais do rotor contra um espectro logarítmico amplo de rigidez de suporte. Ajuda a escolher a rigidez alvo dos mancais.</p>
            <h4>Parâmetros principais:</h4>
            <ul>
                <li><b>Rigidez mín./máx.:</b> limites da rigidez do mancal, em expoente (10^x N/m). Por exemplo, 4 é 10^4 N/m e 10 é 10^10 N/m.</li>
                <li><b>Nº de modos:</b> quantos modos de vibração são resolvidos e desenhados.</li>
            </ul>
        `,
        },
    },
    freq_response: {
        title: {
            en: `<i class='fas fa-wave-square'></i> Frequency Response Help`,
            pt: `<i class='fas fa-wave-square'></i> Ajuda: Resposta em Frequência`,
        },
        body: {
            en: `
            <p>Calculates the steady-state harmonic response of the assembly across a frequency range, plotting both amplitude and phase changes triggered by direct force excitations.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Start/End Speed:</b> Frequency sweep interval boundaries [rad/s].</li>
                <li><b>Input Probes:</b> Node positions and specific degrees of freedom (DoF) where external harmonic forces are applied.</li>
                <li><b>Output Probes:</b> Node positions and specific degrees of freedom (DoF) monitored by displacement sensors to render the output response.</li>
            </ul>
        `,
            pt: `
            <p>Calcula a resposta harmônica em regime permanente ao longo de uma faixa de frequência, mostrando amplitude e fase sob excitação por força direta.</p>
            <h4>Parâmetros principais:</h4>
            <ul>
                <li><b>Velocidade inicial/final:</b> limites da varredura em frequência [rad/s].</li>
                <li><b>Sondas de entrada:</b> nós e graus de liberdade onde as forças harmônicas externas são aplicadas.</li>
                <li><b>Sondas de saída:</b> nós e graus de liberdade monitorados pelos sensores de deslocamento que compõem a resposta.</li>
            </ul>
        `,
        },
    },
    modes: {
        title: {
            en: `<i class='fas fa-project-diagram'></i> Vibration Modes Help`,
            pt: `<i class='fas fa-project-diagram'></i> Ajuda: Modos de Vibração`,
        },
        body: {
            en: `
            <p>Performs a modal eigenvalue analysis at a static rotor speed to extract and display the specific deflected shape (mode shape) of the shaft structure.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Shaft Speed:</b> Constant operational speed at which the gyroscopic and stiffness matrices are evaluated [rad/s].</li>
                <li><b>Nº Modes:</b> Total number of modal points to resolve.</li>
                <li><b>Mode Index:</b> The exact index of the mode shape to render on screen (0 represents the 1st natural mode, 1 the 2nd mode, etc.).</li>
                <li><b>Plot Type:</b> Toggles visualization layout format between 2D or 3D isometric views.</li>
            </ul>
        `,
            pt: `
            <p>Faz a análise modal a uma rotação fixa, para extrair e desenhar a forma deformada (modo de forma) da estrutura do eixo.</p>
            <h4>Parâmetros principais:</h4>
            <ul>
                <li><b>Velocidade do eixo:</b> rotação constante em que as matrizes giroscópica e de rigidez são avaliadas [rad/s].</li>
                <li><b>Nº de modos:</b> quantos modos resolver.</li>
                <li><b>Índice do modo:</b> qual modo desenhar (0 é o 1º modo natural, 1 o 2º, e assim por diante).</li>
                <li><b>Tipo de gráfico:</b> alterna entre a vista 2D e a isométrica 3D.</li>
            </ul>
        `,
        },
    },
    unbalance: {
        title: {
            en: `<i class='fas fa-balance-scale-left'></i> Unbalance Response Help`,
            pt: `<i class='fas fa-balance-scale-left'></i> Ajuda: Resposta ao Desbalanceamento`,
        },
        body: {
            en: `
            <p>Simulates the synchronous dynamic behavior of the rotor under forces generated by physical mass eccentricities distributed along the rotor profile.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Start/End Speed:</b> Rotational speed scanning interval [rad/s].</li>
                <li><b>Unbalance Excitations:</b> Target node containing the unbalance property, specifying its magnitude [kg.m] and spatial phase offset [rad].</li>
                <li><b>Measurement Probes:</b> Target nodes and DoFs where virtual probes measure amplitude outputs.</li>
            </ul>
        `,
            pt: `
            <p>Simula o comportamento síncrono do rotor sob as forças geradas por excentricidades de massa distribuídas ao longo dele.</p>
            <h4>Parâmetros principais:</h4>
            <ul>
                <li><b>Velocidade inicial/final:</b> intervalo de rotação varrido [rad/s].</li>
                <li><b>Excitações de desbalanceamento:</b> nó onde está o desbalanceamento, com magnitude [kg.m] e fase [rad].</li>
                <li><b>Sondas de medição:</b> nós e graus de liberdade onde as sondas virtuais medem a amplitude.</li>
            </ul>
        `,
        },
    },
    time_response: {
        title: {
            en: `<i class='fas fa-hourglass-half'></i> Time Response Help`,
            pt: `<i class='fas fa-hourglass-half'></i> Ajuda: Resposta no Tempo`,
        },
        body: {
            en: `
            <p>Integrates the system equations of motion step-by-step over a timeline, letting you test generic, transient, or custom non-harmonic forces.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Rot. Speed:</b> Constant rotational speed during the physical simulation window [rad/s].</li>
                <li><b>Max Time / Time Steps:</b> Total duration [s] and discretization grid density of the simulation timeline.</li>
                <li><b>Applied Forces F(t):</b> Algebraic formula establishing forces. You can write equations using 't' (time), 'speed', or numpy functions (e.g., <code>1000 * np.cos(speed * t)</code>).</li>
                <li><b>Measurement Probes:</b> Target nodes and DoFs mapped by virtual probes.</li>
                <li><b>Plot Type:</b> Choose between 1D Time histories, 2D Trajectories/Orbits, 3D orbits, or DFFT spectral signatures.</li>
            </ul>
        `,
            pt: `
            <p>Integra as equações de movimento passo a passo ao longo do tempo, o que permite testar forças genéricas, transitórias ou não harmônicas.</p>
            <h4>Parâmetros principais:</h4>
            <ul>
                <li><b>Velocidade de rotação:</b> rotação constante durante a simulação [rad/s].</li>
                <li><b>Tempo máx. / passos de tempo:</b> duração total [s] e densidade da discretização.</li>
                <li><b>Forças aplicadas F(t):</b> fórmula que define a força. Você pode escrever equações com 't' (tempo), 'speed' ou funções do numpy (ex.: <code>1000 * np.cos(speed * t)</code>).</li>
                <li><b>Sondas de medição:</b> nós e graus de liberdade acompanhados pelas sondas virtuais.</li>
                <li><b>Tipo de gráfico:</b> histórico 1D no tempo, trajetórias/órbitas 2D, órbitas 3D ou espectro por DFFT.</li>
            </ul>
        `,
        },
    },
    static: {
        title: {
            en: `<i class='fas fa-compress-arrows-alt'></i> Static Analysis Help`,
            pt: `<i class='fas fa-compress-arrows-alt'></i> Ajuda: Análise Estática`,
        },
        body: {
            en: `
            <p>Evaluates structural static deflections, internal shear stresses, and bending moments due to dead-weight (gravity) and fixed static boundary conditions.</p>
            <h4>Key Parameters:</h4>
            <ul>
                <li><b>Plot Type:</b> Selects the diagram layer representation format: Free Body Diagram, Static Deformation curve, Shearing Force profile, or Bending Moment distribution.</li>
            </ul>
        `,
            pt: `
            <p>Avalia as deflexões estáticas, os esforços cortantes e os momentos fletores devidos ao peso próprio (gravidade) e às condições de contorno estáticas.</p>
            <h4>Parâmetros principais:</h4>
            <ul>
                <li><b>Tipo de gráfico:</b> escolhe o que desenhar — diagrama de corpo livre, curva de deformação estática, perfil de esforço cortante ou distribuição de momento fletor.</li>
            </ul>
        `,
        },
    },
};


// The helpEntry in the current language. A helpEntry that does not exist gives
// back a title built from the category name, as before.
function helpEntry(key) {
    const entry = HelpContent[key];
    const language = currentLanguage();
    if (!entry) {
        return { title: key + ' &mdash; ' + t('help'), body: t('docsSoon') };
    }
    return {
        title: entry.title[language] || entry.title.en,
        body: entry.body[language] || entry.body.en,
    };
}

// Which helpEntry is open. The language change has to know: without this, an
// open help panel stayed in the previous language until it was closed and
// reopened.
let openHelpEntry = null;

export function reapplyHelp() {
    if (openHelpEntry) showHelpEntry(openHelpEntry);
}

function showHelpEntry(key) {
    openHelpEntry = key;
    const content = helpEntry(key);
    document.getElementById('help-modal-title').innerHTML = content.title;
    document.getElementById('help-modal-body').innerHTML = content.body;
    document.getElementById('help-modal-overlay').style.display = 'flex';
}


// Functions for opening modals
export function openGeneralHelp() {
    showHelpEntry('general');
}

export function openAnalysisHelp() {
    showHelpEntry('analysis');
}

export function openSectionHelp(category) {
    showHelpEntry(category);
}

export function closeHelpModal() {
    openHelpEntry = null;
    document.getElementById('help-modal-overlay').style.display = 'none';
}

// Function to intercept the click on the card and trigger contextual help
export const openAnalysisCardHelp = function(event, type) {
    event.stopPropagation();
    openSectionHelp(type);
};

// Close the modal by clicking outside the white box

document.getElementById('help-modal-overlay').addEventListener('click', function(e) {
    if (e.target === this) {
        closeHelpModal();
    }
});
